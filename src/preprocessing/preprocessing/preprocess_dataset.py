import os
import numpy as np
import pandas as pd
import pickle
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from audiolib import AudioDataset, AudioFile

from .preprocess import audio_to_spectrogram


def filter_instrument_ids(filters_root):
    """
    Returns a list of instrument file prefixes, based on
    instrument filters located in the root directory
    """
    print("Loading instrument ID filters...")
    # Load data filters
    root = Path(filters_root)
    data_filters = os.listdir(root)
    data_filters.remove('.DS_Store')
    # Store instrument files to keep in a dict
    instr_filtered = {}
    for file_name in data_filters:
        df_filter = pd.read_csv(root/file_name)
        instr_name = file_name.split('-')[0]
        instr_filtered[instr_name] = (
            list(df_filter.loc[df_filter['keep'] == 1, 'instrument'].values))
    # Return list of instrument IDs to keep in dataset
    print("Done")
    return instr_filtered


def save_spectrograms(
    dataset_root: str,
    instr_name: str,
    instr_ids: list,
    config: dict = None,
    save_path: str = '../data/processed/spectrograms'
):
    """
    Load and filter audio files for an instrument, then save them as spectrograms

    Parameters
    ----------
    dataset_root: str
        Path to the dataset of audio .wav files
    instr_name: str
        Name of the instrument (should match file prefixes)
    instr_ids: list
        IDs of the instruments to apply preprocessing to (should match file prefixes)
    config: dict
        Preprocessing options:

        - `start`: int - Start time for the audio, in seconds (default: 0)
        - `end`: int - End time for the audio, in seconds (default: -1)
        - `time_intervals`: int - Number of audio splits for the FFT (default: 1)
        - `resolution`: int - Frequency resolution, in bins per note (default: 5)
        - `exclude`: list - Qualities to exclude (see metadata `qualities_str`)
    save_path: str
        Where to save the spectrograms (root directory)
    """
    # Paths
    dataset_root = Path(dataset_root)
    save_path = Path(save_path)
    root_save_path = save_path / dataset_root.name / instr_name
    # Parse FFT parameters and save to config.json
    if config is None:
        config = {}
    default_config = {
        'time_intervals': 1,
        'resolution': 5,
        'start': 0,
        'end': -1,
        'exclude': []
    }
    for k, v in default_config.items():
        config[k] = config.get(k, v)
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f)
    # Iterables
    dataset = AudioDataset(dataset_root)
    num_files = sum(
        [len(dataset.file_names_nested[instr_name][id]) for id in instr_ids]
    )
    success_count = 0
    fail_count = 0

    # Loop
    print('\nGenerating {} {} spectrograms:'.format(num_files, instr_name))
    for instr_id in instr_ids:
        instr_id_save_path = root_save_path/instr_id
        os.makedirs(instr_id_save_path, exist_ok=True)
        for file_name in dataset.file_names_nested[instr_name][instr_id]:
            file_save_path = instr_id_save_path/(file_name+'.spec')
            print('- File {} - {} successes / {} failures'.format(
                     file_name, success_count, fail_count),
                  end ='')
            try:
                af = dataset.load_file(file_name)
                if not set(af.info['qualities_str']).intersection(config['exclude']):
                    spec = audio_to_spectrogram(af.audio, config)
                    with open(file_save_path, 'wb') as f:
                        pickle.dump(spec, f)
                    success_count += 1
            except Exception as exc:
                print(' - Last error:', exc, end='')
                fail_count += 1
            print('', end='\r')
                
    print('\nFinished. Saved spectrograms to {}'.format(root_save_path))
