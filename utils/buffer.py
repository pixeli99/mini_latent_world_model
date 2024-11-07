import torch
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, sequence_length):
        self.buffer = deque(maxlen=buffer_size)
        self.sequence_length = sequence_length
        
    def add(self, obs, action):
        self.buffer.append((obs, action))
        
    def sample(self, batch_size):
        # Sample sequence of consecutive transitions
        indices = np.random.randint(0, len(self.buffer) - self.sequence_length, size=batch_size)
        obs_batch = []
        action_batch = []
        
        for idx in indices:
            obs_seq = []
            action_seq = []
            for i in range(self.sequence_length):
                obs, action = self.buffer[idx + i]
                obs_seq.append(obs.numpy())
                action_seq.append(action)
            obs_batch.append(obs_seq)
            action_batch.append(action_seq)
            
        return np.array(obs_batch), np.array(action_batch)
