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

import re
import os
import torch
from IPython.display import display, Audio
import json

import torch.multiprocessing as mp
from transformers import Wav2Vec2Processor, HubertModel, HubertConfig
import soundfile as sf
import joblib
import json
import argparse
import time

# Filepaths
dir_path = os.path.abspath(__file__)
out_path = os.path.dirname(dir_path)

# Initialise parser and multiprocessing
parser = argparse.ArgumentParser()
parser.add_argument('--splits', type=str, nargs='*', default=['test', 'train', 'val'])
parser.add_argument('--batches', type=int, default=1)
mp.set_start_method('spawn', force=True)

# Hubert configs and model
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
config = HubertConfig.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

# Hubert km model
km_100_path = os.path.join(out_path, "hubert_km/km_100.bin")
km_model = joblib.load(open(km_100_path, "rb"))
km_model.verbose = True

# Convert input string tts_utt to list of hubert features
def get_audio(id, tts_utt):
  audio = text2speech(tts_utt)["wav"]
  input_values = processor(audio, return_tensors="pt").input_values

  # Extract hubert features
  hidden_states = model(input_values).last_hidden_state
  discrete_units = km_model.predict(hidden_states.detach().numpy()[0])
  return id, discrete_units.tolist()

def main():
  args = parser.parse_args()

  splits = args.splits
  with torch.no_grad(): 
    for split in splits:
      # Counter for utt id
      counter = 0
      # Maps utt id to list of hubert features
      id_to_hubert = {}
      # Maps utt to utt id
      utt_to_id = {}

      # Multiprocessing batch
      # Stores up to args.batches utts into the batch queue before processing
      batch_queue = []
      batch_size = args.batches
      pool = mp.Pool(processes=3)

      # Open *.history_belief file and save results to *.history_belief_hub file
      with open(f'{out_path}/{split}.history_belief', 'r') as f, \
      open(f'{out_path}/{split}.history_belief_hub', 'w+') as out:
        for line in f:
          # Split by separator tokens (e.g. <|user|>) to get list of dialog turn utts
          utts = re.split(r'<\|.*?\|>', line)
          utts = [x for x in utts if (not x.isspace()) and x != ' \n' and x != '\n' and x != '']

          # Join together word units so utt can be passed through tts
          # (e.g. typical -ly gets changed to typically)
          def to_tts(utt):
            for_tts = re.sub(r'\s+\-','', utt)
            for_tts = for_tts.replace(' s ', 's ')
            return for_tts
          tts_utts = [to_tts(x) for x in utts]
          
          # Go through each dialog turn in the line
          # utt - original utterance, tts_utt - processed utterance to pass to tts
          for utt, tts_utt in zip(utts, tts_utts):
            # If discrete units for utt not already generated, get hubert units
            if utt not in utt_to_id:
              # Increment and generate id for dialog turn
              counter += 1
              audio_id = f'{split}_{str(counter)}'
              utt_to_id[utt] = audio_id

              # Add to batch queue for processing
              batch_queue.append((audio_id, tts_utt))
              # If the queue hits the batch size, process all at once
              if (len(batch_queue) == batch_size):
                # Process queue
                results = pool.starmap(get_audio, batch_queue)
                print(dict(results))
                # Merge dict with id_to_hubert
                id_to_hubert = id_to_hubert | dict(results)
                # Reset queue
                batch_queue = []

            # Get audio id from utt_to_id
            audio_id = utt_to_id[utt]
            # Add audio id in front of original utt
            line = line.replace(utt, f' [{audio_id}]{utt}')
          
          # Write line with audio ids to file
          print(line)
          out.write(line)

      # Process remaining utts in the queue
      results = pool.starmap(get_audio, batch_queue)
      print(dict(results))
      id_to_hubert = id_to_hubert | dict(results)
      batch_queue = []
      # Save id_to_hub dictionary to file
      with open(f'{out_path}/{split}.id_to_hub', 'w+') as link:
        link.write(json.dumps(id_to_hubert))
      
if __name__ == "__main__":
    main()