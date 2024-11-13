import torch
import matplotlib.pyplot as plt
import os
from metadrive import MetaDriveEnv
from models.rssm import RSSM
from models.encoder import ImageEncoderResnet
from models.decoder import ImageDecoderResnet
from utils.preprocessing import preprocess_obs
from metadrive.policy.expert_policy import ExpertPolicy
from config import CONFIG

from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv

import imageio
from utils.buffer import ReplayBuffer
from utils.preprocessing import preprocess_obs
from config import CONFIG
import numpy as np

def inference(model_path, num_episodes=5):
    # Load trained models
    encoder = ImageEncoderResnet(
        in_channels=3,
        depth=32,
        blocks=0,
        resize='stride',
        minres=4
    ).to(CONFIG['device'])

    decoder = ImageDecoderResnet(
        shape=(128, 128, 3),
        input_dim=512 + 256,
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

    # Load model weights from the checkpoint
    checkpoint = torch.load(model_path, map_location=CONFIG['device'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    rssm.load_state_dict(checkpoint['rssm_state_dict'])
    print("Model loaded successfully!")

    # map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM, 
    #         BaseMap.GENERATE_CONFIG: 3,  # 3 block
    #         BaseMap.LANE_WIDTH: 3.5,
    #         BaseMap.LANE_NUM: 2}
    # Initialize environment
    # env = MetaDriveEnv(dict(
    #     use_render=False,
    #     num_scenarios=1,
    #     start_seed=0,
    #     log_level=50,
    #     map_config=map_config,
    #     traffic_density=0.2,
    #     agent_policy=ExpertPolicy
    # ))
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
    # CONFIG['sequence_length'] = 6
    buffer = ReplayBuffer(1000, CONFIG['sequence_length'])
    
    for episode in range(1):
        obs, *_ = env.reset(seed=77)
        terminated = False
        # env.current_track_agent.expert_takeover = True
        policy = env.engine.get_policy(env.current_track_agent.name)
        for _ in range(3):
            obs, _, terminated, truncated, _ = env.step([0, 3])
        
        for _ in range(20):
            # Preprocess the observation
            # action = policy.act(env.current_track_agent.name)
            policy_action = policy.get_action_info()
            velocity = policy_action["velocity"]
            angular_velocity = policy_action["angular_velocity"]
            policy_action = np.array(velocity.tolist() + [angular_velocity])

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
            buffer.add(obs_rgb, policy_action, done=terminated)
            obs, _, terminated, truncated, _ = env.step([0, 3])
        state = rssm.initial_state(batch_size=1)
        images = []
        _step = 0
        while not terminated and _step < 1:
            _step += 1
            obs_batch, action_batch = buffer.sample(1)
            
            obs_batch = torch.FloatTensor(obs_batch).to(CONFIG['device'])
            action_batch = torch.FloatTensor(action_batch).to(CONFIG['device'])
            
            with torch.no_grad():
                # Encode the observation
                embed = encoder(obs_batch[:, :1].flatten(0, 1))
                action_seq = action_batch

                # Get current posterior state using the observation
                is_first_seq = torch.zeros((1, CONFIG['sequence_length']), dtype=torch.bool, device=CONFIG['device'])
                is_first_seq[:, 0] = True
                is_first_seq = is_first_seq[:, :1]
                prev_actions = torch.cat([torch.zeros_like(action_seq[:, :1]), action_seq[:, :-1]], dim=1)
                prev_actions = prev_actions[:, :1, :]
                post, prior = rssm.observe(embed.unsqueeze(0), prev_actions, is_first_seq, state)
                
                feats = torch.cat([post['deter'], post['stoch']], dim=-1)
                # Decode the next state to get the predicted observation for t+1
                decoder_dist = decoder(feats.flatten(0, 1))
                recon_sample = decoder_dist.mean().cpu().permute(0, 2, 3, 1).numpy()[0]
                
                post = {k: v.squeeze(1) for k, v in post.items()}
                
                # Predict the next state using `imagine`
                next_prior = rssm.imagine(action_seq, post)
                next_feats = torch.cat([next_prior['deter'], next_prior['stoch']], dim=-1)
                
                # Decode the next state to get the predicted observation for t+1
                decoder_dist = decoder(next_feats.flatten(0, 1))
                next_recon_sample = decoder_dist.mean().cpu().permute(0, 2, 3, 1).numpy()[0]

            # Visualize the original and predicted next step images
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(obs_rgb.cpu().squeeze().permute(1, 2, 0).numpy())
            axes[0].set_title('Original Observation (t)')
            axes[1].imshow(recon_sample)
            axes[1].set_title('Rec Observation (t)')
            axes[2].imshow(next_recon_sample)
            axes[2].set_title('Predicted Observation (t+1)')
            plt.savefig(f"demo_{episode}.jpg")
            
            
            # fig.canvas.draw()
            # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # images.append(image)
            # plt.close(fig)

            # Simulate an action (random policy for demonstration)
            obs, _, terminated, truncated, _ = env.step([0, 3])

    env.close()
    # imageio.mimsave('demo.gif', images, fps=8)

if __name__ == "__main__":
    inference("/18940970966/mini_latent_world_model/checkpoints/checkpoint_iter_2000.pth")