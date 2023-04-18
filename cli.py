import os

from AIAU_models import *
from AIAU_sampling import *
from AIAU_util import *

import gc
import torch

from einops import rearrange

from audio_diffusion.utils import PadCrop
from audio_diffusion.utils import Stereo
from pydub import AudioSegment
from pydub.utils import make_chunks
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def upscale_audio(input_file, output_file, ckpt_path='rpsmai_diffusion_musdb18hq_ins_v1.ckpt', sampler='adaptive', steps=12, noise_level=0.6, chunk_length=131072, sample_rate=44100):

    # Configurable models and cache folders.

    modelsfolder = os.path.realpath('models/')
    workingpath = os.path.realpath('cache/')

    if not os.path.dirname(ckpt_path):
        ckpt_path = os.path.join(modelsfolder, ckpt_path)

    # Create sampler

    sampler_type = get_sampler(sampler) #fast hq normal adaptive
    
    #Create Processing File
    
    proc_folder = os.path.join(workingpath, "no_vocals_"+sampler_type+"_output/")
    if not os.path.exists(proc_folder):
      os.makedirs(proc_folder)
    else:
      for filename in os.listdir(proc_folder):
        os.remove(os.path.join(proc_folder, filename))
    
    myaudio = AudioSegment.from_file(input_file, "wav") 
    chunk_length_ms = ((chunk_length) / sample_rate) *1000 * 1 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of 65535 samples 
    for i, chunk in enumerate(chunks): 
        chunk_name = os.path.join(proc_folder, f'{i}.wav')
        print ("exporting", chunk_name) 
        chunk.export(chunk_name, format="wav")

    # create model fn
    model_fn = create_model(ckpt_path, sample_rate=sample_rate, sample_size=chunk_length)
    
    for filename in sorted(os.listdir(proc_folder)):
      if not filename.endswith("_upsc.wav") and not filename.endswith("_upsc-final.wav") and not filename.endswith("_upsc-final2.wav") and not filename.endswith("item_0.wav"):
        f = os.path.join(proc_folder, filename)
        if os.path.isfile(f):
          torch.cuda.empty_cache()
          gc.collect()
        
          augs = torch.nn.Sequential(
            PadCrop(chunk_length, randomize=True),
            Stereo()
          )
        
          fp = f
          print("Diffusing Chunk "+fp)
          audio_sample = load_to_device(fp, sample_rate, device)
          audio_sample = augs(audio_sample).unsqueeze(0).repeat([1, 1, 1])
          generated = resample(model_fn, audio_sample, steps, sampler_type, noise_level=noise_level, chunk_length=chunk_length)
        
          print("Saving")
          save_upsc_audio(rearrange(generated, 'b d n -> d (b n)'), fp+"_upsc.wav", sample_rate)
          print("Done")
    
    finalupscoutput = AudioSegment.empty()
    for filename in sorted((os.listdir(proc_folder))):
      if filename.endswith("_upsc.wav"):
        f = os.path.join(proc_folder, filename)
        if os.path.isfile(f):
          finalupscoutput = finalupscoutput.append(AudioSegment.from_file(f, format="wav"), crossfade=0 if not len(finalupscoutput) else 10)

    for filename in os.listdir(proc_folder):
      os.remove(os.path.join(proc_folder, filename))

    
    finalupscoutput.export(output_file, format="wav")
    return finalupscoutput

if __name__ == '__main__':
  # argparse short params
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', help='Input File', required=True)
  parser.add_argument('-o', '--output', help='Output File', required=True)
  parser.add_argument('-c', '--ckpt', help='Checkpoint Path', required=False, default='rpsmai_diffusion_musdb18hq_ins_v1.ckpt')
  parser.add_argument('-s', '--sampler', help='Sampler', required=False, default='adaptive')
  parser.add_argument('-t', '--steps', help='Steps', required=False, default='12')
  parser.add_argument('-n', '--noise', help='Noise Level', required=False, default='0.6')
  args = parser.parse_args()
  upscale_audio(args.input, args.output, ckpt_path=args.ckpt, sampler=args.sampler, steps=int(args.steps), noise_level=float(args.noise))