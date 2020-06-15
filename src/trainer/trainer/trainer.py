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
        path=os.path.join(args['job_dir'], 'logs'),
        hparams=hparams
    )
    callbacks = [tb_writer]
    # Data
    train_loader, test_loader = load_data(args['data_dir'], args['instruments'])
    # Train
    model.fit(
        train_loader,
        test_loader,
        epochs=args['epochs'],
        callbacks=callbacks)
    # Save
    model.save(args['job_dir'])
