import pytorch_modules_2

import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class AutoEncoder:
    def __init__(self,
                 channels_mult=1,
                 bottleneck=16,
                 batch_norm=True,
                 valid_size=0.3,
                 seed=None,
                 aug_params=None,
                 num_workers=0,
                 batch_size=32,
                 device='cpu',
                 lr=0.003,
                 num_epochs=10,
                 log_file='log.csv',
                 weights_file='weights.pt',
                 train_net=True,
                 preload=None):

        self.valid_size = valid_size
        self.seed = seed
        self.aug_params = aug_params
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.log_file = log_file
        self.weights_file = weights_file
        self.preload = preload
        self.train_net = train_net

        self.net = pytorch_modules_2.ECGAutoEncoder(channels_mult=channels_mult,
                                                    bottleneck=bottleneck,
                                                    batch_norm=batch_norm)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=0.0)

    def fit(self, ecgs):

        if self.preload is not None:
            self.net = torch.load(self.preload)

        if self.train_net:

            train, valid = train_test_split(ecgs, test_size=self.valid_size, random_state=self.seed)

            train_ds = PytorchDS(train, aug_params=self.aug_params)
            valid_ds = PytorchDS(valid, aug_params=None)

            train_dl = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   shuffle=True)

            valid_dl = torch.utils.data.DataLoader(valid_ds,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   shuffle=True)

            log = list()
            best_loss = 1e9
            for epoch in range(self.num_epochs):
                train_log = self.run_epoch(train_dl, train=True)
                with torch.no_grad():
                    valid_log = self.run_epoch(valid_dl, train=False)

                row = dict()
                row['epoch'] = epoch
                row.update({f'train_{key}': val for key, val in train_log.items()})
                row.update({f'valid_{key}': val for key, val in valid_log.items()})
                log.append(row)
                pd.DataFrame(log).to_csv(self.log_file)

                print(pd.DataFrame(log))
                if row['valid_loss'] < best_loss:
                    best_loss = row['valid_loss']
                    torch.save(self.net, self.weights_file)

        self.net = torch.load(self.weights_file)

    def transform(self, ecgs):
        self.net.cpu()
        self.net.eval()
        bottlenecks = list()
        ecgs = torch.from_numpy(ecgs).view(-1, 2, 128).float()
        steps = int(np.ceil(ecgs.shape[0] / self.batch_size))
        with torch.no_grad():
            for step in range(steps):
                start = step * self.batch_size
                end = start + self.batch_size
                out = self.net(ecgs[start: end])

                bottlenecks.append(out['bottleneck'].detach().cpu().numpy()[:, :, 0])
        return np.concatenate(bottlenecks)

    def run_epoch(self, dl, train=True):
        if train:
            self.net.train()
        else:
            self.net.eval()

        self.net.to(self.device)

        epoch_loss = 0
        plotted = False
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            ecg_raw = batch['ecg_raw'].to(self.device)
            ecg_aug = batch['ecg_aug'].to(self.device)

            out = self.net(ecg_aug)
            loss = self.criterion(out['autoencoded'][:, :, 16:-16], ecg_raw[:, :, 16:-16])
            epoch_loss += loss.item()
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if not plotted and ecg_raw.shape[0] >= 3:
                plotted = True
                plt.figure(figsize=(20, 10))
                for i in range(3):
                    ecg_in_raw = ecg_raw.cpu().numpy()[i]
                    ecg_in_aug = ecg_aug.cpu().numpy()[i]
                    ecg_out = out['autoencoded'].detach().cpu().numpy()[i]

                    plt.subplot(3, 2, 2 * i + 1)
                    plt.plot(ecg_in_raw[0])
                    plt.plot(ecg_out[0])
                    plt.plot(ecg_in_aug[0], alpha=0.5)
                    plt.ylim(-10, 10)

                    plt.subplot(3, 2, 2 * i + 2)
                    plt.plot(ecg_in_raw[1])
                    plt.plot(ecg_out[1])
                    plt.plot(ecg_in_aug[1], alpha=0.5)
                    plt.ylim(-10, 10)
                plt.show()

        epoch_loss /= len(dl)

        return {'loss': epoch_loss}


class PytorchDS(torch.utils.data.Dataset):
    def __init__(self, ecgs, aug_params=None):
        self.ecgs = ecgs
        self.aug_params = aug_params

    def __len__(self):
        return self.ecgs.shape[0]

    def __getitem__(self, index):
        ecg = self.ecgs[index].reshape((2, -1)).astype('float32')
        result = dict()
        result['ecg_raw'] = ecg
        result['ecg_aug'] = self.augment(ecg.copy())

        return result

    def augment(self, ecg):
        if self.aug_params is not None:
            if np.random.random() < self.aug_params['zero_chan_p']:
                chan = np.random.choice([0, 1])
                ecg[chan] = 2 * (np.random.random(ecg[chan].shape) - 0.5) * self.aug_params['zero_chan_noise']
            ecg += 2 * (np.random.random(ecg.shape) - 0.5) * self.aug_params['noise']
        return ecg
