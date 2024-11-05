import argparse

import hparams
import utils
import multiresunet_model

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
  args = argparse.ArgumentParser()

  args.add_argument('Path',metavar='path',type=str,help='Path to DSD100 pickled spectrograms. See preprocess_data.py for more details')
  args.add_argument('Source',metavar='source',type=str,help='Desired source to separate')
  args.add_argument('Spectrum',metavar='spectrum',type=str,help='Low (lf) or High (hf) frequencies training')
  args.add_argument('Outpath',metavar='model_out_path',type=str,help='Path to save the model to')
  
  ### Parse Args ###
  args = args.parse_args()
  path = args.Path
  source = args.Source
  spectrum = args.Spectrum
  output_path = args.Outpath
  
  ### Load Data ###
  x = np.load(path + 'mixture_' + spectrum + '.npy')
  y = np.load(path + source + '_' + spectrum + '.npy')

  ### Construct model ###
  model = multiresunet_model.Steminator((hparams.frequency_bins,hparams.chunk_size,hparams.n_channels))
  optimizer = tf.keras.optimizers.Adam(lr = hparams.learning_rate)
  model.compile(optimizer, loss='mean_absolute_error')

  ### Training ###
  model.fit(x,y,epochs = hparams.epochs, batch_size = hparams.batch_size)

  ### Save model ###
  model.save(output_path + source + '_' + spectrum + '.h5')




