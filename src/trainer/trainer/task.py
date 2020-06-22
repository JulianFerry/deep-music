import argparse
import json
from pathlib import Path
from . import trainer


def json_loads(s):
    """Handle json sent by gcloud ai platform submit command"""
    s = s.replace('\n', '').replace('$', '')
    return json.loads(s)


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
        '--data_config',
        help='Data config: preprocessing config + subset of instruments to classify',
        type=json_loads,
        default='{}'
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
    args = get_args()
    trainer.train_and_evaluate(args)
