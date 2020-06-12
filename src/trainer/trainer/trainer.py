import numpy as np
import pickle
import os
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
from torchvision import transforms

from .model import Model
from .dataset import load_data


class SummaryWriterCallback(SummaryWriter):

    def on_batch_end(batch, train_loss):
        self.add_scalar('batch_loss/train', train_loss, batch)

    def on_epoch_end(epoch, logs):
        """
        Log
        """
        if logs.get('')
        train_loss, train_acc = validation(model, train_loader)
        self.add_scalar('epoch_loss/train', train_loss, epoch)
        self.add_scalar('epoch_acc/train', train_acc, epoch)
        print('train_loss: %.4f - train_accuracy: %.4f' % (train_loss, train_acc))
        val_loss, val_acc = validation(model, train_loader)
        self.add_scalar('epoch_loss/val', val_loss, epoch)
        self.add_scalar('epoch_acc/val', val_acc, epoch)
        print('val_loss: %.4f - val_accuracy: %.4f' % (val_loss, val_acc))
    
    def on_train_end():
        self.close()


def train_and_evaluate(args):
    """
    Train and evaluate model

    Parameters
    ----------
    args: dict
        Arguments parsed by task.py when the package is called with command-line

    """
    # Model definitino
    model = MusicNet()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.compile(criterion, optimizer)
    # Callbacks
    tb_writer = SummaryWriterCallback(os.path.join(args['job_dir'], 'logs'))
    callbacks = [tb_writer]
    # Data
    train_loader, test_loader = load_data(args['data_dir'], args['instruments'])
    # Train
    model.fit(
        train_loader,
        test_loader,
        epochs=args['epochs'],
        callbacks=callbacks
    # Save
    model.save(args['job_dir'])
