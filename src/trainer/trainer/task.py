import argparse
from pathlib import Path
from . import trainer


def list_loads(s):
    l = s.strip('[]').replace(' ', '').split(',')
    return l


def get_args():
    """
    Argument parser

    Returns
    -------
    Dictionary of arguments

    """
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument(
        '--lr',
        help='Optimiser learning rate',
        type=int,
        default=0.001
    )
    parser.add_argument(
        '--epochs',
        help='Epochs to run the training job for',
        type=int,
        default=1
    )
    parser.add_argument(
        '--instruments',
        help='Instruments to classify in the model',
        type=list_loads,
        default='[*]'
    )

    # Paths
    parser.add_argument(
        '--job_dir',
        help='GCS location to write logs and checkpoint model weights',
    )
    parser.add_argument(
        '--data_dir',
        help='GCS location to fetch data from',
    )

    # Parse
    args = parser.parse_args()
    args = args.__dict__

    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()
    # Run the training job
    trainer.train_and_evaluate(args)
