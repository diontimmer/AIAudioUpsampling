import subprocess
import os
import torch, torchaudio
import IPython.display as ipd
from pydub import AudioSegment
import numpy as np
from typing import Iterable, Tuple

Audio = Tuple[int, np.ndarray]

def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)

def gitclone(url, targetdir=None):
    if targetdir:
        res = subprocess.run(['git', 'clone', url, targetdir], stdout=subprocess.PIPE).stdout.decode('utf-8')
    else:
        res = subprocess.run(['git', 'clone', url], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipi(modulestr):
    res = subprocess.run(['pip', 'install', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipie(modulestr):
    res = subprocess.run(['git', 'install', '-e', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)


#

def save_upsc_audio(audio,filename,sr):
    upscaudio = ipd.Audio(audio.cpu().clamp(-1, 1), rate=sr)
    upscaudio = AudioSegment(upscaudio.data, frame_rate=sr, sample_width=2, channels=2)
    upscaudio.export(out_f=filename, format="wav")

  
def load_to_device(path, sr, device):
    audio, file_sr = torchaudio.load(path)
    if sr != file_sr:
      audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
    audio = audio.to(device)
    return audio


def get_one_channel(audio_data, channel):
  '''
  Takes a numpy audio array and returns 1 channel
  '''
  # Check if the audio has more than 1 channel 
  if len(audio_data.shape) > 1:
    is_stereo = True      
    if np.argmax(audio_data.shape)==0:
        audio_data = audio_data[:,channel] 
    else:
        audio_data = audio_data[channel,:]
  else:
    is_stereo = False

  return audio_data

def get_sampler(mainvariable_diffusionmode):
  if mainvariable_diffusionmode == "fast": # ["v-iplms", "k-heun", "k-lms", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive"]
    sampler_type = "k-dpm-fast"
  elif mainvariable_diffusionmode == "hq":
    sampler_type = "k-dpm-2"
  elif mainvariable_diffusionmode == "normal":
    sampler_type = "v-iplms"
  elif mainvariable_diffusionmode == "adaptive":
    sampler_type = "k-dpm-adaptive"
  else:
    sampler_type = "k-dpm-adaptive"
  return sampler_type

#


def combine_audio(*audio_iterable: Iterable[Audio]) -> Audio:
    """Combines an iterable of audio signals into one."""
    max_len = max([x.shape for _, x in audio_iterable])
    combined_audio = np.zeros(max_len, dtype=np.int32)
    for _, a in audio_iterable:
        combined_audio[:a.shape[0]] = combined_audio[:a.shape[0]] * .5 + a * .5
    return combined_audio


def generate_from_audio(file_path: str, *audio_iterable: Iterable[Audio]):
    sample_rate = audio_iterable[0][0]
    combined_audio = combine_audio(*audio_iterable)
    tensor = torch.from_numpy(
        np.concatenate(
            [
                combined_audio.reshape(1, -1),
                combined_audio.reshape(1, -1)
            ],
            axis=0,
        )
    )
    global recording_file_path
    recording_file_path = file_path
    torchaudio.save(
        file_path,
        tensor,
        sample_rate=sample_rate,
        format="wav"
    )
    return (sample_rate, combined_audio), file_path