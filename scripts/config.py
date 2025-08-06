import os
import torch
from accelerate import Accelerator

def default_config():
    return {
        'project': 'polygon-coloring',
        'run_name': os.getenv('RUN_NAME', 'run1'),
        'epochs': 20,
        'lr': 1e-4,
        'batch_size': 16,
        'img_size': 128,
        'in_ch': 3,
        'out_ch': 3,
        'base_ch': 64,
        'color_embed_dim': 32,
        'data_root': 'data/',
        'lpips_weight': 0.5,
        'colors': ['red','green','blue','yellow','cyan','magenta','purple','orange'],
    }

CONFIG = default_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACCELERATOR = Accelerator()