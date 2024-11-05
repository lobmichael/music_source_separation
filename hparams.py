import librosa

### Audio Hyperparameters ###
sr = 44100

lf_params = {
  'min_f': librosa.note_to_hz('c0'),
  'max_f': 4100,
  'bins_per_octave': 24,
  'gamma': 20
}


hf_params = {
  'min_f': 4100,
  'max_f': 16350,
  'bins_per_octave': 96,
  'gamma': 0
}

### Network Hyperparameters ###

n_channels = 1
chunk_size = 512
frequency_bins = 192
batch_size = 32
learning_rate = 0.0001
epochs = 35
inference_batch_size = 4