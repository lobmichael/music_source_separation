import argparse

import hparams
import utils
import multiresunet_model
import preprocess_data

import tensorflow as tf
import numpy as np
import librosa
import torchaudio
import torch

if __name__ == '__main__':
  args = argparse.ArgumentParser()

  args.add_argument('Path',metavar='path',type=str,help='Path to audio track to be separated')
  args.add_argument('Source',metavar='source',type=str,help='Desired source to separate')
  args.add_argument('Model_path', metavar='path_to_model',type=str,help='Path to saved models')
  args.add_argument('Output_path', metavar='output_path',type=str,help='Output path for separated audio')


  ### Parse args ###
  args = args.parse_args()
  path_to_audio = args.Path
  source = args.Source
  path_to_model = args.Model_path
  output_path = args.Output_path + source + '.wav' 

  ### Load models ###
  model_lf = tf.keras.models.load_model(path_to_model + source + '_lf.h5')
  model_hf = tf.keras.models.load_model(path_to_model + source + '_hf.h5')

  ### Load audio track ###
  y, sr = librosa.load(path_to_audio, hparams.sr, mono = True)

  ### Perform CQT transform on the audio ###
  C_lf,dc_lf,nf_lf = preprocess_data.forward_transform(y,hparams.lf_params['min_f'],hparams.lf_params['max_f'],hparams.lf_params['bins_per_octave'], hparams.lf_params['gamma'])
  C_hf,dc_hf,nf_hf = preprocess_data.forward_transform(y,hparams.hf_params['min_f'],hparams.hf_params['max_f'],hparams.hf_params['bins_per_octave'], hparams.hf_params['gamma'])
  
  dc_lf[:] = 0
  dc_hf[:] = 0
  nf_lf[:] = 0
  nf_hf[:] = 0

  phase_lf = np.angle(C_lf)
  phase_hf = np.angle(C_hf)

  ### Batch Input ###
  c_lf = preprocess_data.make_chunks(C_lf)
  c_hf = preprocess_data.make_chunks(C_hf)

  ### Separate LF and HF ###
  c_lf = model_lf.predict(c_lf,batch_size = hparams.inference_batch_size)
  c_hf = model_hf.predict(c_hf,batch_size = hparams.inference_batch_size)

  ### Reshape Model Output ###
  mag_lf = np.hstack(c_lf[:,:,:,0])[:,:phase_lf.shape[-1]]
  mag_hf = np.hstack(c_hf[:,:,:,0])[:,:phase_hf.shape[-1]]
  c_lf = mag_lf * np.math.e**(phase_lf*1j)
  c_hf = mag_hf * np.math.e**(phase_hf*1j)

  ### Inverse CQT transform using the mixture phase information ###
  y_lf_hat = preprocess_data.backward_transform(c_lf,dc_lf,nf_lf,y.shape[0],hparams.lf_params['min_f'],hparams.lf_params['max_f'],hparams.lf_params['bins_per_octave'], hparams.lf_params['gamma'])
  y_hf_hat = preprocess_data.backward_transform(c_hf,dc_hf,nf_hf,y.shape[0],hparams.hf_params['min_f'],hparams.hf_params['max_f'],hparams.hf_params['bins_per_octave'], hparams.hf_params['gamma'])
  y_hat = y_lf_hat + y_hf_hat

  print(mag_lf.shape)
  print(mag_hf.shape)

  print(y_lf_hat.shape)
  print(y_hf_hat.shape)

  torchaudio.save(output_path, torch.from_numpy(np.expand_dims(y_hat,0)), hparams.sr)





