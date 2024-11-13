import torch
import cv2

def preprocess_obs(obs):
    obs = torch.FloatTensor(obs).permute(2, 0, 1) / 255.0
    return obs

