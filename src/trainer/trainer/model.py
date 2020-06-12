import numpy as np
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import gsutil


class MusicNet(nn.Module):

    def __init__(self):
        super.__init__()
        self.pool = nn.MaxPool1d(2, 2)
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.fc1 = nn.Linear(116 * 32, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 1x476 in, 16x236 out
        x = self.pool(F.relu(self.conv2(x)))  # 16x236 in, 32x116 out
        x = x.view(-1, 32*116)                # flatten each mini-batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

    def compile(criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(
        train_loader,
        val_loader=None,
        epochs=1,
        verbose=1,
        validation_freq=1,
        callbacks=[]
    ):
        """

        """
        if verbose == 1:
            callbacks.append(PrintCallback())
        # Training loop
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            for cb in callbacks:
                cb.on_epoch_start(epoch)
            # Forward pass and backprop
            for batch, data in enumerate(train_loader, 0):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # Batch end
                with torch.no_grad():
                    train_loss *= batch / (batch + 1)
                    train_loss += loss.item() / (batch + 1)
                    train_acc *= batch / (batch + 1)
                    train_acc += evaluate(outputs, labels) / (batch + 1)
                    logs = {'batch_loss/train': train_loss,
                            'batch_acc/train': train_acc}
                    batch_num = epoch * len(train_loader) + batch + 1
                    for cb in callbacks:
                        cb.on_batch_end(batch_num, train_loss)
            # Epoch end
            with torch.no_grad():
                logs = {'epoch_loss/train': train_loss,
                        'epoch_acc/train': train_acc}
                if (val_loader is not None) and (epoch % validation_freq == 0):
                    logs['epoch_loss/val'], logs['epoch_acc/val'] = self.validate(val_loader)
                for cb in callbacks:
                    cb.on_epoch_end(epoch + 1, logs)
        # Train end
        for cb in callbacks:
            cb.on_train_end()

    def evaluate(self, outputs, labels):
        """

        """
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy = equality.type(torch.FloatTensor).mean()
        return accuracy

    def validate(self, test_loader):
        """

        """
        loss = 0
        accuracy = 0
        for n, (inputs, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            # Loss
            outputs = self.forward(inputs)
            loss += self.criterion(outputs, labels).item()
            # Accuracy
            accuracy += self.evaluate(outputs, labels)
        loss /= n+1
        accuracy /= n+1
        return loss, accuracy

    def save(self, path):
        """

        """
        if path.startswith('gs://'):
            path_tmp = Path('/root/train-output')
        else:
            path_tmp = path
        os.makedirs(path_tmp, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path_tmp, 'model.pt'))
        if path.startswith('gs://'):
            gsutil.upload(path_tmp/'*', path])
        print('Saved model to:', path)


class PrintCallback:

    def __init__(self):
        pass

    def on_train_start(self):
        pass

    def on_epoch_start(self, epoch):
        print(f'Epoch {epoch}:')

    def on_batch_end(self, batch, logs):
        print('Batch %d', end ='')
        _print_logs('batch', 'train', end='\r')

    def on_epoch_end(self, epoch, logs):
        print('\nEpoch finished:')
        for dataset in ['train', 'val']:
            _print_logs('epoch', dataset, end='\n')
    
    def on_train_end(self):
        print('Training finished.')

    def _print_logs(self, stage, dataset, end):
        # Loss
        key = f'{stage}_loss/{dataset}'
        if logs.get(key):
            print(f'{dataset}_loss: {logs[key]:.3f}', end=''))
        # Accuracy
        key = f'{stage}_acc/{dataset}'
        if logs.get(key):
            print(f' - {dataset}_accuracy: {logs[key]:.3f}', end=''))
        print('', end=end)

