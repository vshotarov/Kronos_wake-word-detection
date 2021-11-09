import torch
from torch import nn
import torchaudio
import string
import csv
import random
import librosa
import os


class Preprocessor(torchaudio.transforms.MelSpectrogram):
    def __init__(self):
        super(Preprocessor, self).__init__(n_mels=81, win_length=160, hop_length=80)

class Augment(nn.Module):
    def __init__(self):
        super(Augment, self).__init__()

        self.augment = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=70))

    def forward(self, x):
        if torch.rand(1, 1).item() > .5:
            return self.augment(x)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, ambience_csv_path="ambience.csv", validation=False,
            add_ambience_probability=0.5, wav_files_path="resampled_audio",
            wake_label="hey kronos", stop_label="stop"):
        super(Dataset, self).__init__()

        self.wav_files_path = wav_files_path
        self.wake_label = wake_label
        self.stop_label = stop_label
        self.validation = validation
        self.data_table = []
        with open(csv_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for i, row in enumerate(csv_reader):
                if i > 0:
                    self.data_table.append(row)

        self.num_real_samples = len(self.data_table)
        self.ambience_table = []
        self.add_ambience_probability = add_ambience_probability if not validation else 0
        if not validation:
            with open(ambience_csv_path, newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                for i, row in enumerate(csv_reader):
                    if i > 0:
                        self.data_table.append(row)
                        self.ambience_table.append(row)

        self.preprocessor = Preprocessor()

        if validation:
            self.process = nn.Sequential(self.preprocessor)
        else:
            self.process = nn.Sequential(
                    self.preprocessor,
                    Augment())

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, idx):
        audio, _bytes, label = self.data_table[idx]
        waveform, _ = torchaudio.load(os.path.join(self.wav_files_path, audio))

        # Normalize the waveform
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 8000,
                [["gain", "-n"]])

        if not self.validation:
            if torch.rand(1,1) < .3:
                time_shift_percentage = [-1,1][random.randint(0,1)] * random.randint(200,500)
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 8000,
                        [["pitch", str(time_shift_percentage)]])
            elif torch.rand(1,1) < .3:
                pitch_shift_percentage = [-1,1][random.randint(0,1)] * random.randint(25,150)
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, 8000,
                        [["pitch", str(pitch_shift_percentage)], ["rate", "8000"]])

        if idx < self.num_real_samples:
            # This is a real sample, i.e. not an ambience one
            if len(waveform[0]) < 24000:
                waveform = torch.cat((waveform[0], torch.zeros(24000 - len(waveform[0])))).unsqueeze(0)

            waveform = waveform[0][:24000].clone()

            if torch.rand(1, 1).item() < self.add_ambience_probability:
                ambience_audio, _, _ = self.ambience_table[
                        random.randint(0, len(self.ambience_table)-1)]
                ambience_waveform, _ = torchaudio.load(os.path.join(
                    self.wav_files_path, ambience_audio))

                start_point = random.randint(0, ambience_waveform.shape[1]-1-24000)
                ambience_waveform = ambience_waveform[0][start_point:start_point+24000].clone()
                
                waveform = (waveform + ambience_waveform) * .5
        else:
            start_point = random.randint(0, waveform.shape[1]-1-24000)
            waveform = waveform[0][start_point:start_point+24000].clone()

        spectrogram = self.process(waveform)

        if label == self.wake_label:
            label_vector = [0]
        elif label == self.stop_label:
            label_vector = [1]
        else:
            label_vector = [2]

        return spectrogram, label_vector

def pad(device, data):
    spectrograms = []
    labels = []
    for (spectrogram, label) in data:
        spectrograms.append(spectrogram.unsqueeze(0).to(device))
        labels.append(torch.Tensor(label).long().to(device))

    spectrograms = torch.cat(spectrograms)
    labels = torch.cat(labels)

    return spectrograms, labels
