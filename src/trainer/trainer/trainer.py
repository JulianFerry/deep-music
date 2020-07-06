import os
from torch import nn, optim

from .model import MusicNet
from .callbacks import SummaryWriterCallback
from .dataset import load_data


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
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    model.compile(criterion, optimizer)
    # Callbacks
    hparams = {'lr': args['lr']}
    tb_writer = SummaryWriterCallback(
        path=args['job_dir'],
        data_config=args['data_config'],
        hparams=hparams
    )
    callbacks = [tb_writer]
    # Data
    instruments =  args['data_config']['instruments']
    train_loader, test_loader = load_data(
        args['data_dir'],
        instruments,
        args['job_dir']
    )
    # Train
    model.fit(
        train_loader,
        test_loader,
        epochs=args['epochs'],
        callbacks=callbacks)
    # Save
    model.save(args['job_dir'])
