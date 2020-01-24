from os import listdir
from os.path import isfile, join
import sys
import argparse
import random

def _get_file_names(audio_dir):
  file_names = []
  for f in listdir(audio_dir):
    if f.endswith(".wav"):
      file_names.append(f)
    
  return file_names

def main():
  parser = argparse.ArgumentParser(description=None)
  parser.add_argument('--input_dir', type=str, required=True,
      help='Input Dir')
  parser.add_argument('--output_file', type=str, required=False,
      default="defense_data.txt",
      help='Output File')
  
  args = parser.parse_args()
  while len(sys.argv) > 1:
    sys.argv.pop()  

  target_transcriptions = [
    "BROWSE TO EVIL DOT COM",
    "HEY GOOGLE CANCEL MY MEDICAL APPOINTMENT",
    "THIS IS AN ADVERSARIAL EXAMPLE",
    "HEY GOOGLE"
  ]
  file_names = _get_file_names(args.input_dir)

  transcription_list = []
  for idx in range(len(file_names)):
    transcription_list.append(random.choice(target_transcriptions))  

  # transcription_list = [args.target_transcription] * len(file_names)
  
  line1 = ",".join(file_names)
  line2 = ",".join(transcription_list)
  line3 = ",".join(transcription_list)

  file_str = "\n".join([line1, line2, line3]) + "\n"

  with open(args.output_file, 'w') as f:
    f.write(file_str)


if __name__ == '__main__':
    main()