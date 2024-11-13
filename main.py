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
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv

import matplotlib.pyplot as plt
import os

import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter

log_dir = "logs"
writer = SummaryWriter(log_dir=log_dir)

os.environ['SDL_VIDEODRIVER']='dummy'

CONFIG['loss_scales'] = {
    'dyn': 0.5,     # Scale for dynamics loss
    'rep': 0.1,     # Scale for representation loss
    'recon': 1.0    # Scale for reconstruction loss
}

def compute_loss(model, encoder, decoder, obs_seq, action_seq, is_first_seq, config, show_img=False):
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
    if show_img:  # You can set a condition to visualize only occasionally
        batch_size = feats.size(0)
        indices = random.sample(range(batch_size), 4)
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 30))
        
        # 获取 decoder 的均值输出
        recon_samples = decoder_dist.mean().detach().cpu().permute(0, 2, 3, 1).numpy()
        
        # 展平的观测序列
        obs_samples = obs_seq.detach().cpu().permute(0, 1, 3, 4, 2).numpy()
        
        for i, idx in enumerate(indices):
            # 获取重构图像
            recon_sample = recon_samples[idx]
            
            # 获取原始图像样本
            # 假设我们取第一个时间步的观测进行对比
            obs_sample = obs_samples[idx // CONFIG['sequence_length'], idx % CONFIG['sequence_length']]

            # 显示重构图像和原始图像
            axes[i, 0].imshow(recon_sample)
            axes[i, 0].set_title('Reconstructed')
            axes[i, 1].imshow(obs_sample)
            axes[i, 1].set_title('Ground Truth')
        
        plt.savefig('demo.jpg')
        plt.close()

    return total_loss, losses

def train_world_model(resume_from_checkpoint=None):
    # Initialize environment
    # map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM, 
    #         BaseMap.GENERATE_CONFIG: 3,  # 3 block
    #         BaseMap.LANE_WIDTH: 3.5,
    #         BaseMap.LANE_NUM: 2}
    # map_config["config"]=3
    # env = MetaDriveEnv(dict(
    #     use_render=False,
    #     num_scenarios=1,
    #     start_seed=0,
    #     log_level=50,
    #     map_config=map_config, traffic_density=0.2,
    #     agent_policy=ExpertPolicy
    # ))
    # nuscene envs
    
    nuscenes_data=AssetLoader.file_path("/18940970966/nuplan_mini", "meta_drive", unix_style=False) 
    env = ScenarioEnv(
        {
            "reactive_traffic": False,
            "use_render": False,
            "agent_policy": ReplayEgoCarPolicy,
            "data_directory": nuscenes_data,
            "num_scenarios": 1800,
        }
    )
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
        units=512,
        action_dim=CONFIG['action_size'],
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
    global_iter = 0
    for episode in range(1800000):
        obs, *_ = env.reset(seed=77)
        terminated = False
        
        # env.current_track_agent.expert_takeover = True
        policy = env.engine.get_policy(env.current_track_agent.name)
        # policy_action = policy.act(env.current_track_agent.name)

        all_f = []
        while not terminated:
            # obs_rgb = env.render(
            #     mode="topdown", 
            #     window=False,
            #     screen_record=True,
            #     film_size=(512, 512),
            #     screen_size=(128, 128),
            #     draw_contour=False,
            #     scaling=None,
            #     target_agent_heading_up=True,
            #     num_stack=0,
            # )
            obs_rgb = env.render(
                mode="topdown", 
                window=False,
                screen_record=True,
                film_size=(1600, 1600),
                screen_size=(128, 128),
                scaling=4,
                draw_contour=False,
                num_stack=0,
                target_agent_heading_up=True
            )
            obs_rgb = preprocess_obs(obs_rgb)
            policy_action = policy.get_action_info()
            velocity = policy_action["velocity"]
            angular_velocity = policy_action["angular_velocity"]
            policy_action = np.array(velocity.tolist() + [angular_velocity])
            for _ in range(1): # 1 step
                obs, reward, terminated, truncated, info = env.step([0, 3])
            # Preprocess observation and add to buffer
            buffer.add(obs_rgb, policy_action, done=terminated)
        if episode <= 2: continue # Skip the first few episodes to fill the buffer
        for iter in range(200):
            global_iter += 1
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
                obs_batch, action_batch, is_first_seq, CONFIG,
                show_img=(global_iter + 1) % 200 == 0
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
            
            if (global_iter + 1) % 2000 == 0:
                save_path = os.path.join('./checkpoints', f"checkpoint_iter_{global_iter + 1}.pth")
                torch.save({
                    'rssm_state_dict': rssm.state_dict(),
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'iteration': global_iter + 1
                }, save_path)
                print(f"Checkpoint saved at {save_path}")

            writer.add_scalar('Loss/Total', loss.item(), global_iter)
            for key, value in losses_dict.items():
                writer.add_scalar(f'Loss/{key}', value.item(), global_iter)

            print(f"Iter {global_iter}, Total Loss: {loss.item()}")
                
    env.close()

if __name__ == "__main__":
    train_world_model("/18940970966/mini_latent_world_model/checkpoints/checkpoint_iter_76000.pth")