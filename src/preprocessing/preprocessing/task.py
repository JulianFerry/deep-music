import argparse
import json
from pathlib import Path
from warnings import warn
from .batch import filter_instrument_ids, save_spectrograms

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
        '--data_dir',
        help='Path to the dataset of audio .wav files',
        type=str
    )
    parser.add_argument(
        '--filters_dir',
        help='Path to the instrument filter files',
        type=str
    )
    parser.add_argument(
        '--job_dir',
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
    parser.add_argument(
        '--instruments',
        help='Instruments to apply preprocessing to',
        type=json.loads,
        default='[]'
    )

    # Parse
    args = parser.parse_args()
    args = args.__dict__

    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()

    instrument_ids = filter_instrument_ids(args['filters_dir'])
    for instr in args['instruments']:
        if instrument_ids.get(instr):
            save_spectrograms(
                args['data_dir'],
                instr,
                instrument_ids[instr],
                args['config'],
                args['job_dir']
            )
        else:
            warn(f'{instr} not in instrument filters, skipping...')
