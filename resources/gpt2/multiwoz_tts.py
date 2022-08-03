#@title Choose English model { run: "auto" }
lang = 'English'
tag = 'kan-bayashi/ljspeech_vits' #@param ["kan-bayashi/ljspeech_tacotron2", "kan-bayashi/ljspeech_fastspeech", "kan-bayashi/ljspeech_fastspeech2", "kan-bayashi/ljspeech_conformer_fastspeech2", "kan-bayashi/ljspeech_joint_finetune_conformer_fastspeech2_hifigan", "kan-bayashi/ljspeech_joint_train_conformer_fastspeech2_hifigan", "kan-bayashi/ljspeech_vits"] {type:"string"}
vocoder_tag = "none" #@param ["none", "parallel_wavegan/ljspeech_parallel_wavegan.v1", "parallel_wavegan/ljspeech_full_band_melgan.v2", "parallel_wavegan/ljspeech_multi_band_melgan.v2", "parallel_wavegan/ljspeech_hifigan.v1", "parallel_wavegan/ljspeech_style_melgan.v1"] {type:"string"}

from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

text2speech = Text2Speech.from_pretrained(
    model_tag=str_or_none(tag),
    vocoder_tag=str_or_none(vocoder_tag),
    device="cuda",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

import pathlib
import re
import os
import time
import torch
from IPython.display import display, Audio
from pathlib import Path
import json

import string
from torch import true_divide
from transformers import Wav2Vec2Processor, HubertModel, HubertConfig
from datasets import load_dataset
import soundfile as sf
import joblib
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--splits', type=str, nargs='*', default=['test', 'train', 'val'])

def main():
  # Filepaths
  dir_path = os.path.abspath(__file__)
  out_path = os.path.dirname(dir_path)

  # Hubert configs and model
  processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
  config = HubertConfig.from_pretrained("facebook/hubert-base-ls960")
  model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

  # Hubert km model
  km_100_path = os.path.join(out_path, "hubert_km/km_100.bin")
  km_model = joblib.load(open(km_100_path, "rb"))
  km_model.verbose = True

  # Iterate each line in *.history_belief
  splits = args.splits
  with torch.no_grad(): 
    for split in splits:
      Path(f'{out_path}/{split}_audio').mkdir(parents=True, exist_ok=True)
      # Counter for utt id
      counter = 0
      # Dictionary for utt ids to hubert features
      id_to_hubert = {}
      utt_to_id = {}

      # Open *.history_belief file and save results to *.history_belief_hub file
      with open(f'{out_path}/{split}.history_belief', 'r') as f, \
      open(f'{out_path}/{split}.history_belief_hub', 'w+') as out:
        for line in f:
          # Split by tokens to get dialog turns
          utts = re.split(r'<\|.*?\|>', line)
          utts = [x for x in utts if (not x.isspace()) and x != ' \n' and x != '\n' and x != '']

          # Join word units for tts
          def to_tts(utt):
            for_tts = re.sub(r'\s+\-','', utt)
            for_tts = for_tts.replace(' s ', 's ')
            return for_tts
          tts_utts = [to_tts(x) for x in utts]
          
          # Go through each dialog turn in the line
          for utt, tts_utt in zip(utts, tts_utts):
            audio = ''

            # If not already generated, get tts audio and get hubert units
            if utt not in utt_to_id:
              # Increment and generate id for dialog turn
              counter += 1
              audio_id = f'{split}_{str(counter)}'
              utt_to_id[utt] = audio_id

              # Get audio
              audio = text2speech(tts_utt)["wav"]
              input_values = processor(audio, return_tensors="pt").input_values

              # Extract hubert features
              hidden_states = model(input_values).last_hidden_state
              discrete_units = km_model.predict(hidden_states.detach().numpy()[0])
              id_to_hubert[audio_id] = discrete_units.tolist()

              print(discrete_units)

            audio_id = utt_to_id[utt]
            line = line.replace(utt, f' [{audio_id}]{utt}')
          
          print(line)
          out.write(line)
      
      with open(f'{out_path}/{split}.id_to_hub', 'w+') as link:
        link.write(json.dumps(id_to_hubert))
      
if __name__ == "__main__":
    main()