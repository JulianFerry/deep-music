import argparse
import json
from pathlib import Path
from .preprocess_dataset import filter_instrument_ids, save_spectrograms


def list_loads(s):
    l = s.strip('[]').replace(' ', '').split(',')
    return l


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
        '--dataset_path',
        help='Path to the dataset of audio .wav files',
        type=str
    )
    parser.add_argument(
        '--filters_path',
        help='Path to the instrument filter files',
        type=str
    )
    parser.add_argument(
        '--save_path',
        help='Where to save the spectrograms (root directory)',
        type=str
    )

    # Preprocessing arguments
    parser.add_argument(
        '--config',
        help='Preprocessing options',
        type=json.loads,
        default='{}'
    )

    # Parse
    args = parser.parse_args()
    args = args.__dict__

    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()

    instrument_ids = filter_instrument_ids(args['filters_path'])
    for instr, instr_id_list in instrument_ids.items():
        save_spectrograms(
            args['dataset_path'],
            instr,
            instr_id_list,
            args['config'],
            args['save_path']
        )
