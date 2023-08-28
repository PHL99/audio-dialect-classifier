import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import dataset_utils
import train_utils        


if __name__ == '__main__':
    ### Data ingestion
    # Change dir according to data path
    data_dir = "data"
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv")).dropna(subset=['dialect_region'])
    train_df_path = dataset_utils.dialect_wav_data(train_df, data_dir)

    test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv")).dropna(subset=['dialect_region'])
    test_df_path = dataset_utils.dialect_wav_data(test_df, data_dir)

    ### Data preprocess pipeline
    sample_rate = 16000
    max_ms = 4000
    n_fft = 1024
    n_mels = 128
    n_mfcc = 128
    freq_mask_rate = 0.1
    time_mask_rate = 0.1
    train_transform = nn.Sequential(
        train_utils.RandomEffector(sample_rate, p=0.5),
        train_utils.FixedSignalLength(sample_rate, max_ms),
        T.MFCC(n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels
            }
        ),
        T.FrequencyMasking(freq_mask_rate*n_mfcc),
        T.FrequencyMasking(freq_mask_rate*n_mfcc),
        T.TimeMasking(time_mask_rate*n_mfcc),
        T.TimeMasking(time_mask_rate*n_mfcc)
    )
    test_transform = nn.Sequential(
        train_utils.FixedSignalLength(sample_rate, max_ms),
        T.MFCC(n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels
            }
        )
    )

    ### Data loading pipeline
    batch_size = 32
    train_data = dataset_utils.AudioDataset(train_df_path, new_sample_rate=sample_rate, transforms=train_transform, keep_channel=True)
    test_data = dataset_utils.AudioDataset(test_df_path, new_sample_rate=sample_rate, transforms=test_transform, keep_channel=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    print("Number of train batches:", len(train_loader))
    print("Batched data size:", next(iter(train_loader))[0].size())

    ### Model
    model = nn.Sequential(
        nn.BatchNorm2d(1),
        nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(64, 8),
    )

    ### Training
    start_epoch = 0
    end_epoch = 20
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=int(len(train_loader)),
                                                    epochs=end_epoch-start_epoch, anneal_strategy='linear')

    load_path = "save_state/audio_dialect_classifier_{}_ep.tar".format(start_epoch) if start_epoch > 0 else None
    save_path = "save_state/audio_dialect_classifier_{}_ep.tar".format(end_epoch)

    train_utils.train_and_save(model, loss_func, optimizer, scheduler, (train_loader, test_loader), end_epoch, 
                               save_path, load_path)

