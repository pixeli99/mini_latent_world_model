import torch
import cv2

def preprocess_obs(obs):
    # Resize and normalize observation
    obs = cv2.resize(obs, (128, 128))
    obs = torch.FloatTensor(obs).permute(2, 0, 1) / 255.0
    return obs

