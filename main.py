import torch
import torch.nn.functional as F
from metadrive import TopDownMetaDrive, TopDownSingleFrameMetaDriveEnv, MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from models.rssm import RSSM
from models.encoder import ImageEncoderResnet
from models.decoder import ImageDecoderResnet
from utils.buffer import ReplayBuffer
from utils.preprocessing import preprocess_obs
from config import CONFIG

from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy

import matplotlib.pyplot as plt
import os

from torch.utils.tensorboard import SummaryWriter

log_dir = "logs"
writer = SummaryWriter(log_dir=log_dir)

os.environ['SDL_VIDEODRIVER']='dummy'

CONFIG['loss_scales'] = {
    'dyn': 0.5,     # Scale for dynamics loss
    'rep': 0.1,     # Scale for representation loss
    'recon': 1.0    # Scale for reconstruction loss
}

def compute_loss(model, encoder, decoder, obs_seq, action_seq, is_first_seq, config):
    batch_size = obs_seq.shape[0]

    # Get initial state from the RSSM model
    state = model.initial_state(batch_size)

    # Initialize losses
    losses = {'dyn': 0.0, 'rep': 0.0, 'recon': 0.0}
    scales = config['loss_scales']

    # Encode observations
    embed_seq = encoder(obs_seq.flatten(0, 1)).view(CONFIG['batch_size'], CONFIG['sequence_length'], -1)

    # Prepare previous actions (include the previous action and current action sequence)
    prev_actions = torch.cat([torch.zeros_like(action_seq[:, :1]), action_seq[:, :-1]], dim=1)

    # Run RSSM observe step over the sequence
    post, prior = model.observe(embed_seq, prev_actions, is_first_seq, state)

    # Compute losses
    # Dynamics loss (KL divergence between posterior and prior)
    dyn_loss = model.dyn_loss(post, prior)
    losses['dyn'] = dyn_loss.mean()

    # Representation loss (could be the same as dynamics loss or computed differently)
    rep_loss = model.rep_loss(post, prior)
    losses['rep'] = rep_loss.mean()

    # Decoder reconstruction loss (Negative Log Likelihood)
    # The decoder should output distribution parameters (e.g., mean and std for Gaussian)
    feats = torch.cat([post['deter'], post['stoch']], dim=-1)
    decoder_dist = decoder(feats.flatten(0, 1))
    recon_loss = -decoder_dist.log_prob(obs_seq.flatten(0, 1)).mean()
    losses['recon'] = recon_loss

    # Total loss with scaling
    total_loss = sum(scales[k] * v for k, v in losses.items())

    # Optionally, log or visualize reconstructions
    if True:  # You can set a condition to visualize only occasionally
        recon_sample = decoder_dist.mean().detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        obs_sample = obs_seq.detach().cpu().permute(0, 1, 3, 4, 2).numpy()[0][0]
        fig, axes = plt.subplots(1, 2, figsize=(15, 30))
        axes[0].imshow(recon_sample)
        axes[1].imshow(obs_sample)
        plt.savefig('demo.jpg')
        plt.close()

    return total_loss, losses

def train_world_model(resume_from_checkpoint=None):
    # Initialize environment
    env = MetaDriveEnv(dict(
        use_render=False,
        num_scenarios=1000,
        start_seed=2,
        map="O", traffic_density=0.2,
        agent_policy=ExpertPolicy
    ))
    
    # Initialize models
    encoder = ImageEncoderResnet(
        in_channels=3,  # Update based on your observation shape
        depth=32,
        blocks=0,
        resize='stride',
        minres=4
    ).to(CONFIG['device'])

    # Modify decoder to output distribution parameters
    decoder = ImageDecoderResnet(
        shape=(128, 128, 3),  # Update based on your output shape (C, H, W)
        input_dim=512+256,  # Match with the latent feature size from RSSM
        depth=32,
        blocks=0,
        resize='stride',
        minres=4,
    ).to(CONFIG['device'])
    
    rssm = RSSM(
        deter=512, 
        stoch=256, 
        classes=None, 
        initial="learned", 
        unimix=0.01, 
        action_clip=1.0, 
        winit='normal', 
        fan='avg', 
        units=512
    ).to(CONFIG['device'])

    
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint)
        rssm.load_state_dict(checkpoint['rssm_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Resumed training from checkpoint.")
        
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        list(rssm.parameters()) + 
        list(encoder.parameters()) + 
        list(decoder.parameters()),
        lr=CONFIG['learning_rate']
    )
    
    # Initialize replay buffer
    buffer = ReplayBuffer(CONFIG['buffer_size'], CONFIG['sequence_length'])
    
    # Training loop
    for episode in range(100000):
        obs, *_ = env.reset()
        env.agent.expert_takeover = True
        terminated = False
        
        while not terminated:
            env.current_track_agent.expert_takeover = True
            policy = env.engine.get_policy(env.current_track_agent.name)
            policy_action = policy.act(env.current_track_agent.name)
            
            for _ in range(2):
                obs, reward, terminated, truncated, info = env.step(policy_action)

            # Preprocess observation and add to buffer
            obs_rgb = env.render(
                mode="topdown", 
                window=False,
                screen_record=False,
                film_size=(500, 500),
                screen_size=(128, 128),
                semantic_map=False,
                draw_contour=False,
                target_agent_heading_up=True,
                num_stack=0,
            )
            obs_rgb = preprocess_obs(obs_rgb)
            buffer.add(obs_rgb, policy_action)
        for iter in range(10000000):
            # Sample a batch from the buffer
            obs_batch, action_batch = buffer.sample(CONFIG['batch_size'])
            # Convert to torch tensors
            obs_batch = torch.FloatTensor(obs_batch).to(CONFIG['device'])
            action_batch = torch.FloatTensor(action_batch).to(CONFIG['device'])
            
            # Prepare is_first sequence (indicates whether each sequence starts a new episode)
            is_first_seq = torch.zeros((CONFIG['batch_size'], CONFIG['sequence_length']), dtype=torch.bool, device=CONFIG['device'])
            is_first_seq[:, 0] = True  # Assuming each sequence is from a new episode
            
            # Compute loss and update
            loss, losses_dict = compute_loss(
                rssm, encoder, decoder, 
                obs_batch, action_batch, is_first_seq, CONFIG
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(rssm.parameters()) + 
                list(encoder.parameters()) + 
                list(decoder.parameters()),
                CONFIG['grad_clip']
            )
            optimizer.step()
            
            if (iter + 1) % 100 == 0:
                save_path = os.path.join('./checkpoints', f"checkpoint_iter_{iter + 1}.pth")
                torch.save({
                    'rssm_state_dict': rssm.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'iteration': iter + 1
                }, save_path)
                print(f"Checkpoint saved at {save_path}")

            writer.add_scalar('Loss/Total', loss.item(), iter)
            for key, value in losses_dict.items():
                writer.add_scalar(f'Loss/{key}', value.item(), iter)

            print(f"Iter {iter}, Total Loss: {loss.item()}")
                
    env.close()

if __name__ == "__main__":
    train_world_model()