# Kronos - Wake word detection
The wake word detection module for [Kronos - my personal virtual assistant proof of concept](https://github.com/vshotarov/Kronos).

## Table of Contents
<ol>
<li><a href="#overview">Overview</a></li>
<li><a href="#training">Training</a></li>
<li><a href="#data-augmentation">Data Augmentation</a></li>
<li><a href="#tips">Tips</a></li>
</ol>

## Overview
The wake word detection module is responsible for checking whether a voice recording contains the wake word.

The way it is implemented is as a much smaller version of the [DeepSpeech](https://arxiv.org/pdf/1512.02595.pdf) architecture, but with a classifier at the end, so we classify samples as one of three categories - wake word, stop\*, pass. Where DeepSpeech has ~39m parameters, this model has 366k.

\*The *stop* category, detects a stop word, with the idea that we might want to interrupt the voice assistant. Currently it's not utilized it Kronos.

The inputs to the network are **NOT** the raw audio waveform, but rather a processed version of it and that preprocessor lives in `data.py`. All it does is it converts the audio from an amplitude over time graph to a heatmap of frequencies over time using a [Mel Spectrogram](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.MelSpectrogram). 

The output of the network is unnormalized log probabilities, but you can use the `WWDModel.classify()` method to directly get the class with the largest probability.

During training we also randomly augment a percentage of the samples, in order to provide a bit more variety to the small dataset and help the network build resilience to unimportant variations. Have a look at the [data augmentation](#data-augmentation) section fore more info.

For an overview of how wake word detection fits in the full application have a look at [the main repo](https://github.com/vshotarov/Kronos#overview).

## Training
To train, run `train.py` with the relevant arguments.

```
usage: train.py [-h] [-ps PATH_TO_SAVE_MODEL] [-pw PATH_TO_WAV] [-lw LABEL_WAKE] [-ls LABEL_STOP]
                [-ne NUM_EPOCHS] [-ve VALIDATE_EVERY] [-lrde LEARNING_RATE_DECAY_EVERY]
                [-lrdr LEARNING_RATE_DECAY_RATE]
                train_dataset validation_dataset test_dataset ambience_csv

Kronos virtual assistant - Wake word detection trainer

The dataset .csv files need to have the following columns:
   wav_filename wav_filesize transcript

There's an extra ambience csv file, which is more a list of .wav files than a dataset,
but it's used to provide a bit of additional background ambience to the recorded
samples. The ambience recordings are expected to be just background noise in
the room you're planning to use the virtual assistant in.

positional arguments:
  train_dataset         path to the dataset .csv file for training
  validation_dataset    path to the dataset .csv file for validation during training
  test_dataset          path to the dataset .csv file for testing after training
  ambience_csv          path to the dataset .csv file storing the names of ambience .wav files

optional arguments:
  -h, --help            show this help message and exit
  -ps PATH_TO_SAVE_MODEL, --path_to_save_model PATH_TO_SAVE_MODEL
                        path to save the trained model at. By default it's a file called
                        saved_model.torch in the current directory.
  -pw PATH_TO_WAV, --path_to_wav PATH_TO_WAV
                        path to the directory storing the .wav files specified in the datasets. By
                        default it's 'wav' directory in the current directory.
  -lw LABEL_WAKE, --label_wake LABEL_WAKE
                        the text in the wake samples. By default it's 'hey kronos'.
  -ls LABEL_STOP, --label_stop LABEL_STOP
                        the text in the wake samples. By default it's 'stop'.
  -ne NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        how many epochs of training to run. By default it's 1250.
  -ve VALIDATE_EVERY, --validate_every VALIDATE_EVERY
                        how often to validate in epochs. By default it's every 10.
  -lrde LEARNING_RATE_DECAY_EVERY, --learning_rate_decay_every LEARNING_RATE_DECAY_EVERY
                        how often to decay learning rate. By default it's every 15.
  -lrdr LEARNING_RATE_DECAY_RATE, --learning_rate_decay_rate LEARNING_RATE_DECAY_RATE
                        how much to decay learning rate. By default it's .99.
```

Here's an example of what the dataset .csv files look like:

```
wav_filename,wav_filesize,transcript
hey_kronos_0.wav,32044,hey kronos
hey_kronos_1.wav,32044,hey kronos
stop_0.wav,32216,stop
pass_0.wav,32174,set a timer for five minutes
```

where the `wav_filename` columns contains names relative to the `-pw, --path_to_wav` argument.

## Data Augmentation
During training we perform some data augmentation in order to try and avoid overfitting and force the network to build resilience against unimportant variations.

The single most important process we do is normalize the gain, which massively improves the results by ignoring variations in how loud the utterance was spoken.

After that 30% of the time we perform some time stretching or pitch shifting.

Then, half of the time we pick an ambience sample from the provided as an argument ambience dataset, which gets put as a background to the actual current sample, in order to provide some variety to the state of the surrounding environment.

Lastly after converting the waveform to a spectrogram, half of the time we perform some [time](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.TimeMasking) and [frequency masking](https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.FrequencyMasking) to essentially damage the sample a bit, in order to force the network to build some resilience to recording issues.

## Tips
- use a wake word that's very clear, specific and very rarely spoken otherwise
- consider not using *hey/hi* as a part of your wake word, as I've had to record a lot of samples to overcome the issue of the network thinking I've said *hey kronos*, when in reality I've said *hey, hey bonus (not that I say that often), hey betty, hey mom*, etc.
- when recording samples, it's helpful to use different microphones in different environment settings
