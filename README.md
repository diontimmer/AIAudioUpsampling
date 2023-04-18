
# AI Audio Upsampling

Restoration/Variation of audio using chunked diffusion. Heavily based on Reed+Sam AI's SplitDiffusion implementation of Zach Evans Dance Diffusion. Uses dance diffusion to restore audio in variable chunks.


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
                        Input file to upsample/restore (input.wav)
  -o OUTPUT, --output OUTPUT
                        Output file to render (output.wav)
  -c CKPT, --ckpt CKPT  Checkpoint Path (Upsampler/DD model)
  -s SAMPLER, --sampler SAMPLER
                        Sampler type ([)fast, hq, normal, adaptive])
  -t STEPS, --steps STEPS
                        Amount of diffusion steps to take
  -n NOISE, --noise NOISE
                        Noise Level (0-1) to add to process diffusion over

