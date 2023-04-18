from audio_diffusion.models import DiffusionAttnUnet1D
from copy import deepcopy
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiffusionUncond(nn.Module):
    def __init__(self, global_args):
        super().__init__()
        self.diffusion = DiffusionAttnUnet1D(global_args, n_attn_layers = 4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

def create_model(ckpt_path, sample_rate, sample_size):
    args = Object()
    args.latent_dim = 0
    args.sample_size = sample_size
    args.sample_rate = sample_rate
    print("Creating the model...")
    model = DiffusionUncond(args)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    model = model.requires_grad_(False).to(device)
    model_fn = model.diffusion_ema
    print("Model created")
    # # Remove non-EMA
    del model.diffusion
    return model_fn

class Object(object):
    pass