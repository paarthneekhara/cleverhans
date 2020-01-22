from os import listdir
from os.path import isfile, join
import sys
import argparse

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
  parser.add_argument('--target_transcription', type=str, required=True,
      help='Target transcription')
  
  args = parser.parse_args()
  while len(sys.argv) > 1:
    sys.argv.pop()  


  file_names = _get_file_names(args.input_dir)
  transcription_list = [args.target_transcription] * len(file_names)
  
  line1 = ",".join(file_names)
  line2 = ",".join(transcription_list)
  line3 = ",".join(transcription_list)

  file_str = "\n".join([line1, line2, line3]) + "\n"

  with open("defense_data.txt", 'w') as f:
    f.write(file_str)


if __name__ == '__main__':
    main()