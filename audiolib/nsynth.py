import json
import re
import librosa
import librosa.display
from functools import partial
from pathlib import Path
from scipy.io import wavfile
from warnings import warn

# Fix circular imports with absolute import
from importlib import import_module
pkg = __name__.split('.')[0]
audio_ = import_module(f'{pkg}.audio')


class AudioDataset:

    def __init__(self, path='../data/raw/nsynth-train/'):
        """
        Load the file names and metadata in the specified NSynth directory

        Args:
            path - str or pathlib.Path - path to the dataset folder
        """
        self.path = path if type(path) is Path else Path(path)
        # Metadata
        self.examples = json.load(open(self.path/'examples.json'))
        # File names per instrument
        self.file_names = {}
        self.file_names_nested = {}
        re_instrument = partial(re.findall, r'(^.*?)_\d')
        for file in self.examples.keys():
            prefix = self.examples[file]['instrument_str']
            instrument = re_instrument(prefix)[0]
            (self.file_names.setdefault(instrument, [])
                            .append(file))
            (self.file_names_nested.setdefault(instrument, {})
                                   .setdefault(prefix, [])
                                   .append(file))
        # Unique instruments
        self.unique_instruments = list(self.file_names.keys())
        # File counts
        self.file_counts = {instr: len(self.file_names[instr])
                            for instr in self.unique_instruments}

    def _check_instrument(self, instrument):
        """
        Check that an instrument name (str) exists in the dataset
        """
        if instrument not in self.unique_instruments:
            raise(ValueError('Instrument {} is not one of {}'.format(
                instrument, self.unique_instruments)))

    def load_file(self, file_name=None, instrument=None, file_index=None):
        """
        Returns the path of an audio file in the dataset.
        There are two ways of loading a file path:
            - Either specify file_name to fetch an NSynth file path by name
            - Alternatively, restrict files to those starting with the `instrument` string
              and use `file_index` to reference the file index within that subset of files

        Args:
            file_name - str - name of the audio file
            instrument - str - must match one of the AudioDataset's unique_instruments
            file_index - int - index of the file within that instrument's file_names
        """
        if file_name is not None:
            file = Path(file_name + '.wav')
        elif instrument is not None and file_index is not None:
            self._check_instrument(instrument)
            file = Path(self.file_names[instrument][file_index] + '.wav')
        else:
            raise(ValueError(
                'A file_name or both instrument and file_index must be specified'))
        path = self.path/'audio'/file.name
        info = self.examples[file.stem]
        return AudioFile(path, info)


class AudioFile:
    """
    Loads audio data and metadata from an NSynth dataset.
    Stores audio data as an Audio object, which contains methods for audio analysis.

    Methods:
        __init__ - load audio data from a .wav file
        reload() - reload Audio object to undo any processing applied to it
    """

    def __init__(self, path, info):
        """
        Load audio data from .wav file

        Args:
            path - str or Pathlib.path - path to the audio .wav file
            info - dict - example metadata from the NSynth dataset
        """
        self.path = path
        self.info = info
        if self.info.get('pitch') is None:
            warn('No pitch information found. Some functionality will not work')
        self.reload()

    def reload(self):
        """
        Load audio from the path specified in __init__
        Automatically called on class creation
        """
        sampling_rate, audio = wavfile.read(self.path)
        if self.info.get('pitch'):
            fundamental_freq = librosa.midi_to_hz(self.info['pitch'])
        else:
            fundamental_freq = None
        self.audio = audio_.Audio(audio, sampling_rate, fundamental_freq)
