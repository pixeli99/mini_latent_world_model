import torch

CONFIG = {
    # Model
    'hidden_size': 256,
    'embedding_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Training
    'batch_size': 16,
    'sequence_length': 32,
    'learning_rate': 1e-4,
    'grad_clip': 1000.0,
    'buffer_size': 100000,
    
    # Environment
    'action_size': 2,  # MetaDrive default
    'img_size': (128, 128, 3),  # Resized observation
}

