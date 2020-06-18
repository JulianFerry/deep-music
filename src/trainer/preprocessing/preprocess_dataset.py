import os
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from pathlib import Path
from audiolib import AudioDataset, AudioFile

from .preprocess import audio_to_spectrogram


def load_filtered_instruments(root='../data/interim/filters/train'):
    # Load data filters
    root = Path(root)
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
    instrument_ids = list(instr_filtered.keys())
    return instrument_ids


def save_spectrograms(
    dataset_root: str,
    instr_name: str,
    fft_params: dict = None,
    excluded_qualities: list = ['fast_decay'],
    save_path: str = '../data/processed/'
):
    """
    Load and filter audio files for an instrument, then save them as spectrograms

    Parameters
    ----------
    dataset_root: str
        Path to the dataset of audio .wav files
    instr_name: 
    """
    # Params
    if fft_params is None:
        fft_params = {
            'time_intervals': 1,
            'resolution': 5,
            'start': 0,
            'end': -1
        }
    dataset_root = Path(dataset_root)
    root_save_path = Path(os.path.join(
        save_path,
        dataset_root.name,
        f'time_intervals={fft_params['time_intervals']}',
        f'resolution={fft_params['resolution']}',
        instr_name
    ))
    dataset = AudioDataset(dataset_root)
    num_files = sum(
        [len(dataset.file_names_nested[instr_name][id]) for id in instr_ids]
    )
    instr_ids = instr_filtered[instr_name]
    # Iterables
    success_count = 0
    fail_count = 0

    # Loop
    print('Generating {} {} spectrograms'.format(num_files, instr_name))
    for instr_id in instr_ids:
        instr_id_save_path = save_path/instr_id
        os.makedirs(instr_id_save_path, exist_ok=True)
        for file_name in dataset.file_names_nested[instr_name][instr_id]:
            file_save_path = instr_id_save_path/(file_name+'.spec')
            print('- File {} - {} successes / {} failures'.format(
                     file_name, success_count, fail_count),
                  end ='\r')
            try:
                af = dataset.load_file(file_name)
                if not set(af.info['qualities_str']).intersection(exluded_qualities):
                    spec = audio_to_spectrogram(af.audio, fft_params)
                    with open(file_save_path, 'wb') as f:
                        pickle.dump(spec, f)
                    success_count += 1
            except Exception as exc:
                #print(exc)
                fail_count += 1
                
    print('\nFinished. Saved spectrograms to {}\n'.format(save_path))


if __name__ == "__main__":

    train = AudioDataset(path='../data/raw/nsynth-train/')
    instruments = load_filtered_instruments()
    for instr in instruments:
        save_spectrograms('../data/raw/nsynth-train/', instr, start=0.2, end=3)