
import argparse
from .untar import untar


def get_args():
    """
    Argument parser.
    See preprocessing.preprocess_dataset modoule for more info.

    Returns
    -------
    Dictionary of arguments

    """
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument(
        '--bucket_name',
        help='Cloud storage bucket used to load and save data',
        default='deep-musik-data',
        type=str
    )
    parser.add_argument(
        '--zip_dir',
        help='Bucket path to the tar.gz file\'s parent directory',
        default='download.magenta.tensorflow.org/datasets/nsynth/',
        type=str
    )
    parser.add_argument(
        '--zip_file',
        help='Name of the .tar.gz zip file',
        type=str,
        required=True
    )
    parser.add_argument(
        '--save_dir',
        help='Bucket path to the directory in which to save decompressed files',
        default='data/raw/',
        type=str
    )

    # Parse
    return parser.parse_args().__dict__

if __name__ == "__main__":
    args = get_args()
    untar(**args)