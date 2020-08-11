import os
import pandas as pd
from scipy.io import wavfile
from audiolib import Audio

import apache_beam as beam


def load_audio(f):
    """
    Parameters
    ----------
    f: beam.io.fileio.ReadableFile
        File object returned by beam.ReadMatches()

    """
    file_name = f.metadata.path.split('/')[-1]
    sampling_rate, data = wavfile.read(f.open())
    audio = Audio(data, sampling_rate)
    return file_name, audio


class PreProcessor:

    def __init__(self, fft_params=None):
        """
        Initialise the preprocessor with the parameters used to
        transform the audio

        Parameters
        ----------
        fft_params: dict
            Dictionary with optional keys
            - 'start', 'end' (float: default 0, -1):
              audio start and end time cutoffs in seconds
            - 'time_intervals', 'resolution' (default 1 and 5):
              FFT parameters see audiolib.Audio.to_spectrogram for details

        """
        self.fft_params = {} if fft_params is None else fft_params
        default_params = {
            'time_intervals': 1,
            'resolution': 5,
            'start': 0,
            'end': -1,
            'exclude': []
        }
        for k, v in default_params.items():
            self.fft_params[k] = self.fft_params.get(k, v)
        
    def audio_to_spectrogram(self, record):
        """
        Convert an audio object to a spectrogram given a set of fft_params

        """
        file_name, audio = record
        spec = (audio
            .trim(self.fft_params['start'], self.fft_params['end'])
            .to_spectrogram(
                self.fft_params['time_intervals'],
                self.fft_params['resolution'],
                cqt=True
            ))
        return file_name, spec

    def save_config(self, path):
        try: os.makedirs(path, exist_ok=True)
        except: pass
        (pd.Series(self.fft_params)
        .to_json(os.path.join(path, 'config.json')))