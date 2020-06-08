import argparse
from . import model


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
        '--epochs',
        help='Epochs to run the training job for',
        type=int,
        default=1
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
    model.train_and_evaluate(args)
