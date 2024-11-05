import argparse
import glob

import numpy as np
import librosa
from essentia.standard import (NSGConstantQ, 
    NSGIConstantQ)

import hparams
import utils

def parse_files(path, source):

  if source == 'mixture':
    path = path + 'Mixtures/Dev/*/' + str(source) + '.wav'
    paths = sorted(glob.glob(path))
  else:
    path = path + 'Sources/Dev/*/' + str(source) + '.wav'
    paths = sorted(glob.glob(path))
  return paths

def forward_transform(y, min_f, max_f, bpo, gamma):
  # Parameters
  params = {
            # Backward transform needs to know the signal size.
            'inputSize': y.size,
            'minFrequency': min_f,
            'maxFrequency': max_f,
            'binsPerOctave': bpo,
            # Minimum number of FFT bins per CQ channel.
            'minimumWindow': 4,
            'gamma': gamma
          }


  # Forward and backward transforms
  constantq, dcchannel, nfchannel = NSGConstantQ(**params)(y)

  return constantq, dcchannel, nfchannel

def backward_transform(c, dc, nf, orig_size, min_f, max_f, bpo, gamma):
  # Parameters
  params = {
            # Backward transform needs to know the signal size.
            'inputSize': orig_size,
            'minFrequency': min_f,
            'maxFrequency': max_f,
            'binsPerOctave': bpo,
            # Minimum number of FFT bins per CQ channel.
            'minimumWindow': 4,
            'gamma': gamma
          }


  # Forward and backward transforms
  y = NSGIConstantQ(**params)(c, dc, nf)

  return y


def make_chunks(c):
  cqt = np.abs(c).astype(np.float16)
  cqt = np.asfortranarray(cqt)
  padded_cqt = librosa.util.fix_length(cqt,hparams.chunk_size*np.ceil(cqt.shape[-1]/hparams.chunk_size).astype(int))
  framed_cqt = librosa.util.frame(padded_cqt,hparams.chunk_size,hparams.chunk_size)
  samples = np.transpose(framed_cqt,(2,0,1))
  cqt_input = np.expand_dims(samples,-1)
  return cqt_input

if __name__ == '__main__':
  args = argparse.ArgumentParser()

  args.add_argument('Path',metavar='path',type=str,help='Path to DSD100')
  args.add_argument('Source',metavar='source',type=str,help='Desired source to preprocess for separation. Use mixture to preprocess the mixtures')
  args.add_argument('Output_path',metavar='output_path',type=str,help='Output path for the pikled spectrograms')

  args = args.parse_args()
  path = args.Path
  source = args.Source
  outpath = args.Output_path

  if path[-1] != '/':
    path = path + '/'
  if outpath[-1] != '/':
    outpath = outpath + '/'


  files = parse_files(path, source)
  mag_lf_array = []
  mag_hf_array = []

  for i in range(0,len(files)):
    print(files[i])
    y, sr = librosa.load(files[i], hparams.sr, mono = True)
    C_lf,_,_ = forward_transform(y,hparams.lf_params['min_f'],hparams.lf_params['max_f'],hparams.lf_params['bins_per_octave'], hparams.lf_params['gamma'])
    C_hf,_,_ = forward_transform(y,hparams.hf_params['min_f'],hparams.hf_params['max_f'],hparams.hf_params['bins_per_octave'], hparams.hf_params['gamma'])
    c_lf = make_chunks(C_lf)
    c_hf = make_chunks(C_hf)
    mag_lf_array.append(c_lf)
    mag_hf_array.append(c_hf)
    if  i == 1:
      break


  mag_lf = utils.list_to_array(mag_lf_array)
  mag_hf = utils.list_to_array(mag_hf_array)


  filename_lf = source + '_lf.npy'
  filename_hf = source + '_hf.npy'
  utils.pickle(mag_lf, outpath, filename_lf)
  utils.pickle(mag_hf, outpath, filename_hf)



