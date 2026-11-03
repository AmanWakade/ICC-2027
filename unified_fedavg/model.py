import random
import numpy as np
import torch
from torch import nn

SEED = 42
INPUT_DIM = 54

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

class DGFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def get_weights(model):
    return [v.detach().cpu().numpy().astype(np.float32, copy=True) for v in model.state_dict().values()]

def set_weights(model, weights):
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError('Unexpected number of model arrays')
    state = {}
    for k, a in zip(keys, weights):
        target = model.state_dict()[k]
        if tuple(a.shape) != tuple(target.shape):
            raise ValueError(f'Shape mismatch for {k}: {a.shape} vs {tuple(target.shape)}')
        state[k] = torch.tensor(a, dtype=target.dtype)
    model.load_state_dict(state, strict=True)
