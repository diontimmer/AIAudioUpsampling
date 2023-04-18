from audio_diffusion.models import DiffusionAttnUnet1D
from copy import deepcopy
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models_map = {

    "rpsmai_diffusion_23pop_ins_v1": {'downloaded': False,
                        'sha': "96a9f5a5cebc6871e5eef9506a6ad7b9f2e13b028ff79b81a940f8b1628e6157", 
                        'uri_list': ["https://rpsmai.dev/splitdiffusion/models/rpsmai_diffusion_23pop_ins_v1.ckpt"],
                        'sample_rate': 44100,
                        'sample_size': 131072
                        },
    "rpsmai_diffusion_musdb18hq_ins_v1": {'downloaded': False,
                        'sha': "b4f08e103b89aa1cac33ca094d4b7d531ffecca53edbf1d765e2de375aacca4b", 
                        'uri_list': ["https://rpsmai.dev/splitdiffusion/models/rpsmai_diffusion_musdb18hq_ins_v1.ckpt"],
                        'sample_rate': 44100,
                        'sample_size': 131072
                        },
}

def download_model(diffusion_model_name, uri_index=0):
    if diffusion_model_name != 'custom':
        model_filename = get_model_filename(diffusion_model_name)
        model_local_path = os.path.join(model_path, model_filename)
        if os.path.exists(model_local_path) and check_model_SHA:
            print(f'Checking {diffusion_model_name} File')
            with open(model_local_path, "rb") as f:
                bytes = f.read() 
                hash = hashlib.sha256(bytes).hexdigest()
                print(f'SHA: {hash}')
            if hash == models_map[diffusion_model_name]['sha']:
                print(f'{diffusion_model_name} SHA matches')
                models_map[diffusion_model_name]['downloaded'] = True
            else:
                print(f"{diffusion_model_name} SHA doesn't match. Will redownload it.")
        elif os.path.exists(model_local_path) and not check_model_SHA or models_map[diffusion_model_name]['downloaded']:
            print(f'{diffusion_model_name} already downloaded. If the file is corrupt, enable check_model_SHA.')
            models_map[diffusion_model_name]['downloaded'] = True

        if not models_map[diffusion_model_name]['downloaded']:
            for model_uri in models_map[diffusion_model_name]['uri_list']:
                wget(model_uri, model_local_path)
                with open(model_local_path, "rb") as f:
                  bytes = f.read() 
                  hash = hashlib.sha256(bytes).hexdigest()
                  print(f'SHA: {hash}')
                if os.path.exists(model_local_path):
                    models_map[diffusion_model_name]['downloaded'] = True
                    return
                else:
                    print(f'{diffusion_model_name} model download from {model_uri} failed. Will try any fallback uri.')
            print(f'{diffusion_model_name} download failed.')


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