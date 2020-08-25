import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .callbacks import PrintCallback
from . import gsutil


class MusicNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(2, 2)
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.fc1 = nn.Linear(116 * 32, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2)
        # Send to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        """
        Model forward pass / prediction
        """
        x = self.pool(F.relu(self.conv1(x)))    # 1x476 in, 16x236 out
        x = self.pool(F.relu(self.conv2(x)))    # 16x236 in, 32x116 out
        x = x.view(-1, 32*116)                  # flatten each mini-batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def compile(
        self,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer
    ):
        """
        Set the training loss function and optimiser

        Parameters
        ----------
        loss_fn: torch.nn.modules.loss._Loss
            Loss function for training
        optimizer: torch.optim.Optimizer
            Optimisation method for training

        """
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit(
        self,
        train_loader: torch.utils.data.dataloader.DataLoader,
        valid_loader: torch.utils.data.dataloader.DataLoader = None,
        epochs: int = 1,
        verbosity: int = 1,
        validation_freq: int = 1,
        callbacks: list = None
    ):
        """
        Run training job. Callbacks are called at each stage of training.

        Parameters
        ----------
        train_loader: torch.utils.data.dataloader.DataLoader
            Training data loader
        valid_loader: torch.utils.data.dataloader.DataLoader
            Validation data loader (optional)
        epcohs: int
            Number of epochs to run the training job for
        verbosity: int (0, 1, 2)
            How much of the training logs to print:
            - 0 will not print any progress
            - 1 will print model performance after each epoch
            - 2 will print model performance bar for each batch
        validation_freq: int
            Epoch frequency at which the model's performance on the
            validation set will be calculated and logged
        callbacks: list
            Training callback functions, defined in trainer.callbacks

        """
        if callbacks is None:
            callbacks = []
        callbacks.append(PrintCallback(epochs, len(train_loader), verbosity))
        for cb in callbacks:
            cb.on_train_start()
        # Training loop
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            for cb in callbacks:
                cb.on_epoch_start(epoch + 1)
            # Backpropagation
            for batch, data in enumerate(train_loader, 0):
                for cb in callbacks:
                    cb.on_batch_start(batch + 1)
                # Fetch data from train loader and send to GPU
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                # Run forward pass and backpropagation
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # Batch end - evaluate and log performance
                with torch.no_grad():
                    train_loss *= batch / (batch + 1)
                    train_loss += loss.item() / (batch + 1)
                    train_acc *= batch / (batch + 1)
                    train_acc += self.evaluate(outputs, labels) / (batch + 1)
                    logs = {'batch_loss/train': train_loss,
                            'batch_acc/train': train_acc}
                    batch_total = epoch * len(train_loader) + batch + 1
                    for cb in callbacks:
                        cb.on_batch_end(batch + 1, batch_total, logs)
            # Epoch end - evaluate and log performance on validation set
            with torch.no_grad():
                logs = {'epoch_loss/train': train_loss,
                        'epoch_acc/train': train_acc}
                if (valid_loader is not None) and (epoch % validation_freq == 0):
                    val_loss, val_acc = self.validate(valid_loader)
                    logs['epoch_loss/val'], logs['epoch_acc/val'] = val_loss, val_acc
                for cb in callbacks:
                    cb.on_epoch_end(epoch + 1, logs)
        # Train end
        for cb in callbacks:
            cb.on_train_end()

    def evaluate(self, outputs, labels):
        """
        Calculate the model's accuracy on the given data

        """
        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy = equality.type(torch.FloatTensor).mean()
        return accuracy

    def validate(
        self,
        valid_loader: torch.utils.data.dataloader.DataLoader
    ):
        """
        Calculate loss and accuracy on the valid_loader data

        Parameters
        ----------
        valid_loader: torch.utils.data.dataloader.DataLoader
            Validation data loader

        Returns
        -------
        loss, accuracy: tuple (float, float)

        """
        loss = 0
        accuracy = 0
        for n, (inputs, labels) in enumerate(valid_loader):
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            # Loss
            outputs = self.forward(inputs)
            loss += self.loss_fn(outputs, labels).item()
            # Accuracy
            accuracy += self.evaluate(outputs, labels)
        loss /= n+1
        accuracy /= n+1
        return loss, accuracy

    def save(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            Path to save (local or Google Storage)

        """
        local_path = Path(path.replace('gs://', ''))
        os.makedirs(local_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(local_path, 'model.pt'))
        if path.startswith('gs://'):
            gsutil.upload(os.path.join(local_path, '*'), path)
        print('Saved model to:', path)
