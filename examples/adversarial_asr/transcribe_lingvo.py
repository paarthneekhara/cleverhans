import tensorflow as tf
from lingvo import model_imports
from lingvo import model_registry
import numpy as np
import scipy.io.wavfile as wav
import generate_masking_threshold as generate_mask
from tool import create_features, create_inputs
import time
from lingvo.core import cluster_factory
from absl import app
import argparse
from os import listdir
from os.path import isfile, join
import json
import sys

def _get_file_names(audio_dir):
  file_names = []
  for f in listdir(audio_dir):
    if f.endswith(".wav"):
      file_names.append(f)
    
  return file_names

def _decode_audio(audio_dir, file_name):
  file_path = join(audio_dir, file_name)
  sample_rate_np, audio_temp = wav.read(file_path)

  audios = []
  lengths = []

  if max(audio_temp) < 1:
    audio_np = audio_temp * 32768    
  else:
    audio_np = audio_temp

  length = len(audio_np)

  audios.append(audio_np)
  lengths.append(length)

  

  lengths_freq = (np.array(lengths) // 2 + 1) // 240 * 3
  max_length_freq = max(lengths_freq)
  masks_freq = np.zeros([1, max_length_freq, 80])

  audios_np = np.zeros([1, length])

  audios_np[0, :lengths[0]] = audios[0]
  masks_freq[0, :lengths_freq[0], :] = 1

  return audios_np, sample_rate_np, np.array(["BROWSE TO EVIL DOT COM"]), masks_freq

def main():

  checkpoint = "./model/ckpt-00908156"
  
  parser = argparse.ArgumentParser(description=None)
  parser.add_argument('--dirs', type=str, nargs='+', required=True,
      help='Filepath of original input audio')
  
  

  args = parser.parse_args()
  while len(sys.argv) > 1:
    sys.argv.pop()

  with tf.device("/gpu:0"):
    tf.set_random_seed(1234)
    tfconf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=tfconf) as sess:
      params = model_registry.GetParams('asr.librispeech.Librispeech960Wpm', 'Test')
      params.cluster.worker.gpus_per_replica = 1
      cluster = cluster_factory.Cluster(params.cluster)
      with cluster, tf.device(cluster.GetPlacer()):
        params.vn.global_vn = False
        params.random_seed = 1234
        params.is_eval = True
        model = params.cls(params)
        task = model.GetTask()
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        input_tf = tf.placeholder(tf.float32, shape=[1, None])
        tgt_tf = tf.placeholder(tf.string)
        sample_rate_tf = tf.placeholder(tf.int32) 
        mask_tf = tf.placeholder(tf.float32, shape=[1, None, 80])


        features = create_features(input_tf, sample_rate_tf, mask_tf)
        shape = tf.shape(features)
        inputs = create_inputs(model, features, tgt_tf, 1, mask_tf)

        metrics = task.FPropDefaultTheta(inputs)              
        loss = tf.get_collection("per_loss")[0]  
        
        # prediction
        decoded_outputs = task.Decode(inputs)
        dec_metrics_dict = task.CreateDecoderMetrics()



        for audio_dir in args.dirs:
          file_names = _get_file_names(audio_dir)
          transcriptions = {}
          for fidx, file_name in enumerate(file_names):
            audios_np, sample_rate, tgt_np, mask_freq  = _decode_audio(audio_dir, file_name)

            feed_dict={input_tf: audios_np, 
                 sample_rate_tf: sample_rate, 
                 tgt_tf: tgt_np, 
                 mask_tf: mask_freq}
            
            try:
              losses = sess.run(loss, feed_dict)  
              predictions = sess.run(decoded_outputs, feed_dict)
            except:
              print ("Error in transcribing: ", file_name)
              continue
            
            task.PostProcessDecodeOut(predictions, dec_metrics_dict)
            wer_value = dec_metrics_dict['wer'].value * 100.
            transcriptions[file_name] = predictions['topk_decoded'][0, 0].lower()
            
            print(fidx, "pred-{},{} : {}".format(audio_dir, file_name, predictions['topk_decoded'][0, 0]))
            

          with open(join(audio_dir, "transcriptions.json"), 'w') as f:
            f.write(json.dumps(transcriptions))
            
if __name__ == '__main__':
  main()
