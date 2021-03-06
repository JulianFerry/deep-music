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
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9)
    model.compile(loss_fn, optimizer)
    # Callbacks
    hparams = {
        'lr': args['lr'],
        'data_config': args['train_config']['data_config_id']
    }
    tb_writer = SummaryWriterCallback(
        path=args['job_dir'],
        train_config=args['train_config'],
        hparams=hparams
    )
    callbacks = [tb_writer]
    # Data
    instruments = args['train_config']['instruments']
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
        callbacks=callbacks
    )
    # Save
    model.save(args['job_dir'])
