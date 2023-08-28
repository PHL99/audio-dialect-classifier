import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F



### CSV-Ingestion ###
def find_wav_path(df):
    temp_df_bool = df.iloc[:, -5:-1].sum(axis=1)
    temp_df = df[temp_df_bool == 1.]
    return temp_df[temp_df['is_audio']]


def dialect_wav_data(df, data_dir):
    df_path = find_wav_path(df)[['dialect_region', 'path_from_data_dir']]       # locate label and path audio file
    df_path['path_from_data_dir'] = df_path['path_from_data_dir'].apply(lambda x: os.path.join(data_dir, 'data', x))    # specify full path
    one_hot = pd.get_dummies(df_path['dialect_region'].apply(lambda x: x[-1]).astype(int) - 1)      # get onehot vectors
    return pd.concat([df_path.drop('dialect_region', axis=1), one_hot], axis=1)     # replace label with onehot


### Dataset Pipeline ###
class AudioDataset(Dataset):
    def __init__(self, wav_path_csv, new_sample_rate=16000, transforms=None, keep_channel=False):
        self.wav_path = wav_path_csv
        self.new_sample_rate = new_sample_rate
        self.transforms = transforms
        self.keep_channel = keep_channel
        
    def __len__(self):
        return self.wav_path.shape[0]
    
    def __getitem__(self, idx):
        path = self.wav_path.iloc[idx, 0]
        label = torch.tensor(self.wav_path.iloc[idx, 1:].to_numpy(dtype=np.float32))
        data, sample_rate = torchaudio.load(path)
        if sample_rate != self.new_sample_rate:
            data = F.resample(data, sample_rate, self.new_sample_rate)
        if self.transforms:
            data = self.transforms(data)
        if not self.keep_channel:
            data = data.squeeze(0)
        return (data, label)