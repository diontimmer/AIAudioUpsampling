
# AI Audio Upsampling

Restoration/Variation of audio using chunked diffusion. Heavily based on [Reed+Sam AI's SplitDiffusion]("https://colab.research.google.com/drive/1fkwG7yzy9uegEoMpigKNpA0pmXyzrBAz#scrollTo=--Ml82LICtki") implementation of Zach Evans Dance Diffusion. Uses dance diffusion to restore audio in variable chunks. Big shoutout to Reed+Sam Media for coming up with the original idea, it works really great and its fun to mess with.

This is basically a CLI wrapper for the Reed+Sam notebook so you can whip it out and use it on a bunch of files fast.


## Installation

[Install Torch]("https://pytorch.org/get-started/locally/") for your system; preferably the CUDA version if your system can handle it. You technically only need base torch + audio:
```
Windows:
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

Linux:
pip3 install torch torchaudio
```

After this install the requirements:
```
pip install -r requirements.txt
```


    
## Usage

```
python cli.py ->
  -i INPUT, --input INPUT
                        Input folder to upsample/restore
  -o OUTPUT, --output OUTPUT
                        Output folder to render to
  -c CKPT, --ckpt CKPT  Checkpoint Path (Upsampler/DD model)
  -s SAMPLER, --sampler SAMPLER
                        Sampler type ([)fast, hq, normal, adaptive])
  -t STEPS, --steps STEPS
                        Amount of diffusion steps to take
  -n NOISE, --noise NOISE
                        Noise Level (0-1) to add to process diffusion over

