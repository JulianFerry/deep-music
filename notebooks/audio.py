import json
import re
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
from IPython.display import Audio, display
from warnings import warn
import math

from fft import butterworth_filter, pretty_spectrogram, invert_pretty_spectrogram

# AN EVALUATION OF AUDIO FEATURE EXTRACTION TOOLBOXES

class AudioDataset:
    def __init__(self, path='../data/raw/nsynth-train/', metadata='examples.json'):
        """
        Load names and metadata of audio files in the specified directory
        
        Args:
            dataset - str - Nsynth dataset folder ('train', 'valid', 'test')
            instrument - str - ('keyboard_acoustic', 'guitar_acoustic')
            file_index - int - index of the file within the specified dataset 
        """
        self.path = path if type(path) is Path else Path(path)
        # Metadata
        self.examples = json.load(open(self.path/'examples.json'))
        # File names per instrument
        self.file_names = {}
        self.file_names_nested = {}
        re_instrument = partial(re.findall, '(^.*?)_\d')
        for file in self.examples.keys():
            prefix = self.examples[file]['instrument_str']
            instrument = re_instrument(prefix)[0]
            self.file_names.setdefault(instrument, []).append(file)
            self.file_names_nested.setdefault(instrument, {}).setdefault(prefix, []).append(file)
        # Unique instruments
        self.unique_instruments = list(self.file_names.keys())
        # File counts
        self.file_counts = {instr: len(self.file_names[instr])
                            for instr in self.unique_instruments}
    
    def _check_instrument(self, instrument):
        """Check input"""
        if instrument not in self.unique_instruments:
            raise(ValueError(f'Instrument {instrument} is not one of {self.unique_instruments}'))
        
    def get_filepath(self, instrument=None, file_index=None, file_name=None):
        """
        Returns the path of an audio file in the dataset. Files will be restricted
        to those starting with the `instrument` string - `file_index` is used to
        select the file within that subset of files.
        
        Args:
            instrument - str - one of ('keyboard_acoustic', 'guitar_acoustic')
            file_index - int - index of the file within the instrument's subset of files
        """
        if file_name is not None:
            file = Path(file_name + '.wav')
        elif instrument is not None and file_index is not None:
            self._check_instrument(instrument)
            file = Path(self.file_names[instrument][file_index] + '.wav')
        else:
            raise(ValueError('Either file_name or both instrument and file_index must be specified'))
        path = self.path/'audio'/file.name
        info = self.examples[file.stem]        
        return path, info


class AudioFile:
    """
    Defines methods to load, process and visualise audio data using spectrograms
    
    Methods:
        __init__ - load audio data from a .wav file
        trim_audio - trim start and end times of the audio data
        filter_audio - apply a butterworth filter (lp, hp, bp, bs) to the audio data
        plot_audio - plot the audio waveform in the time domain
        play_audio - create a widget to play that audio
        audio_to_spectrogram - generate an FFT spectrogram from the audio data
        filter_harmonics - filter out all non-harmonic frequencies from the spectrogram
        spectrogram_to_audio - convert a spectrogram back to an audio waveform
        spectrogram_to_log - convert a spectrogram's frequencies to the log domain
        plot_spectrogram - plot a spectrogram
        convolve_spectrogram (experimental) - apply non-continuous 1-D convolution
    """
    
    def __init__(self, path, info):
        """
        Load audio data from .wav file
        
        Args:
            start - int - when to start the audio clip, in seconds
            end - int - when to end the audio clip, in seconds
            butter - tuple - butterworth filter (low, high, btype)
        """
        self.path = path
        self.info = info
        if info.get('pitch'):
            self.fundamental_freq = librosa.midi_to_hz(info['pitch'])
        else:
            warn('No pitch information found\n'
                 'Some funtionality will not work unless you modify .fundamental_freq')
            self.fundamental_freq = None
        self.cqt_parmas = None
        self.reload_audio()
    
    # Audio methods
    def reload_audio(self):
        """
        Load audio from the path specified in __init__
        Automatically called on class creation
        """
        self.sampling_rate, self.audio = wavfile.read(self.path)
        self.audio = self.audio.astype(np.float)
        self.nyquist = self.sampling_rate // 2
        self.duration = self.audio.shape[0] / self.sampling_rate

    def trim_audio(self, start=0, end=-1):
        """
        Trim the audio data to start and end times (in seconds)
        
        Args:
            start - numeric - start time (seconds)
            end - numeric - end time (seconds)
        """
        start_idx = 0
        if start != 0:
            if 0 < start < self.duration:
                start_idx = int(start * self.sampling_rate)
            else:
                warn(f'Start set to 0')
                start_idx = 0
        end_idx = -1
        if end != -1:
            if start < end <= self.duration:
                end_idx = int(end * self.sampling_rate)
            else:
                warn(f'End set to {self.duration}')
                end_idx = -1
        self.audio = self.audio[start_idx:end_idx]
        self.duration = self.audio.shape[0] / self.sampling_rate

    def filter_audio(self, lowcut, highcut, btype='lowpass', order=4):
        """
        Apply butterworth filter to the audio data
        
        Args:
            lowcut - numeric - lowest frequency
            highcut - numeric - highest frequency
            btype - one of ('lowpass', 'highpass', 'bandpass', 'bandstop')
            order - int - (1, 2, 3, 4, 5) complexity of the butterworth filter
        """
        self.audio = butterworth_filter(
            self.audio, self.sampling_rate, lowcut, highcut, btype, order=order)
    
    def plot_audio(self):
        """
        Plot the loaded audio waveform
        """
        time = np.linspace(0., self.duration, self.audio.shape[0])
        plt.figure(figsize=(10, 3))
        plt.plot(time, self.audio)
        plt.xlim(time.min(), time.max())
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)');
        plt.tight_layout()
        plt.show()

    def play_audio(self, autoplay=False):
        """
        Create widget which plays audio
        """
        display(Audio(self.audio, rate=self.sampling_rate, autoplay=autoplay))
    
    # Spectrogram methods
    def audio_to_spectrogram(self, time_intervals=1, resolution=1):
        """
        Args:
            time_intervals: how many time intervals to create a spectrogram for  
        """
        # setting resolution=1 maximises the frequency domain resolution
        fft_size = self._get_fft_size(resolution)
        self.hop_length = int(self.audio.shape[0] / time_intervals)
        # Create spectrogram
        self.spectrogram = librosa.core.stft(
            self.audio,
            n_fft=fft_size,
            hop_length=self.hop_length + 1,
        )

    def _get_fft_size(self, resolution):
        """
        Get fft_size for a given resolution
        
        resolution - float <= 1 - 1 is maximal frequency resolution that is a power of 2
        """
        fft_size = 2**int(np.log2(resolution * self.audio.shape[0]))
        return fft_size
    
    def spectrogram_to_audio(self, spectrogram):
        """
        Convert spectrogram to audio
        """
        f_inverse = librosa.istft if spectrogram.dtype == 'complex64' else librosa.griffinlim
        recovered_audio = f_inverse(
            spectrogram,
            hop_length=self.hop_length + 1,
            center=False
        )
        display(Audio(recovered_audio, rate=self.sampling_rate))

    def filter_harmonics(self, neighbour_radius=0):
        """
        Only keep spectrogram amplitudes for harmonics of the fundamental frequency
        
        Args:
            neighbour_radius - int -  number of neighbouring frequencies to include
        """
        self.spectrogram_harm = np.zeros(self.spectrogram.shape, dtype='complex64')
        step = int(self.fundamental_freq * self.spectrogram.shape[0] / self.nyquist)
        for t in range(self.spectrogram.shape[1]):
            for i in range(neighbour_radius+1):
                self.spectrogram_harm[step+i::step, t] = self.spectrogram[step+i::step, t]
                self.spectrogram_harm[step-i::step, t] = self.spectrogram[step-i::step, t]
    
    def audio_to_cqt_spectrogram(self, resolution=1, min_freq=32.7, max_freq=-1):
        """ Convert
        
        Args:
            spectrogram - spectrogram to convert to log domain
        """
        # Setup
        max_freq = self._get_max_freq(max_freq)
        self._set_cqt_params(resolution, min_freq, max_freq)
        self.spectrogram_cqt = np.abs(librosa.cqt(
            y=self.audio,
            sr=self.sampling_rate,
            **self.cqt_params,
            scale=False,
            sparsity=0,     # high-res but possibly slow
            res_type='fft'  # high-res but possibly slow
        ))[:, :1]

    def _set_cqt_params(self, note_resolution, min_freq, max_freq):
        bins_per_octave = int(12 * note_resolution)
        num_octaves = np.log2(max_freq/min_freq)
        self.cqt_params = {
            'bins_per_octave': bins_per_octave,
            'n_bins': int(num_octaves * bins_per_octave),
            'fmin': min_freq,
            'hop_length': (self.audio.shape[0] // (2**int(num_octaves-1))) * (2**int(num_octaves-1))
        }

    def _get_max_freq(self, max_freq):
        """
        Calculate max frequency
        """
        if max_freq is None:
            max_freq = 10 * self.fundamental_freq
        elif max_freq == -1:
            max_freq = self.nyquist
        return max_freq
    
    def _get_min_freq(self, min_freq, cqt):
        """
        Calculate min frequency
        """
        if min_freq is None:
            if cqt:
                min_freq = self.cqt_params['fmin']
            else:
                min_freq = 0
        return min_freq

    def plot_spectrogram(self, spectrogram, spec_thresh=0, min_freq=None, max_freq=None,
                         title='', figsize=(10, 6), ax=None, y_harm=None, cqt=False,
                         **kwargs):
        """
        Plot a spectrogram
        
        Args:
            spectrogram - np.array - spectrogram to plot
            spec_thresh - minimum spectrogram threshold to plot
            min_freq - numeric - plot minimum frequency
            max_freq - numeric - plot maximum frequency - 'default' uses 10x the fundamental
            title - str - plot title
            figsize - tuple - (width, height)
            ax - matplotlib ax
            y_harm - whether to use harmonic frequencies as y axis labels
            cqt - bool - whether the spectrogram frequencies are on a cqt scale
        """
        if cqt and (self.cqt_params is None):
            raise(ValueError('CQT transform has not yet been called'))
        max_freq = self._get_max_freq(max_freq)
        min_freq = self._get_min_freq(min_freq, cqt)
        plt.figure(figsize=figsize)
        cax = librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spectrogram)).clip(min=spec_thresh),
            sr=self.sampling_rate,
            cmap=plt.cm.afmhot,
            y_axis='cqt_hz' if cqt else 'hz',
            ax=ax,
            bins_per_octave=self.cqt_params['bins_per_octave'] if cqt else None,
            fmin=self.cqt_params['fmin'] if cqt else None,
            hop_length=self.cqt_params['hop_length'] if cqt else None,
            **kwargs
        )
        plt.ylim(min_freq, max_freq)
        plt.title(title)
        plt.colorbar(format='%+2.0f dB')
        # Change yticks
        if y_harm is not None:
            y_min = self.fundamental_freq * ((min_freq // self.fundamental_freq)+1)
            if cqt:
                y_max = self.fundamental_freq * y_harm
                y_step = self.fundamental_freq
            else:
                y_max = self.fundamental_freq * (max_freq // self.fundamental_freq)
                y_step = (y_max - y_min) / 9
            ytick_freqs = range(int(y_min), int(y_max)+1, int(y_step))
            plt.yticks(ytick_freqs)
        plt.show()
    
    # Convolutions
    def convolve_spectrogram(self, spectrogram):
        """
        Unravel the spectrogram, apply 1-D conv to it then 'ravel' it back
        """
        self._get_harmonic_groups(spectrogram)
        self.spectrogram_conv = np.ones(spectrogram.shape)[::2] # * -8
        mask = [1, 1]
        for t in range(spectrogram.shape[1]):
            # Time slice to apply convolution to
            spectrogram_slice = spectrogram[t]
            # Convolve each group of harmonics
            harmonic_convs = []
            for group in self.harmonic_groups:
                harmonic_amplitudes = spectrogram_slice[group]
                harmonic_convs = np.convolve(mask, harmonic_amplitudes, mode='valid')
                self.spectrogram_conv[t, group[:-1]] = harmonic_convs

    def _get_harmonic_groups(self, spectrogram):
        """
        Create groups of harmonic frequencies to unwrap the spectrogram
        """
        max_freq = spectrogram.shape[0]
        self.harmonic_groups = []
        for freq in list(range(1, max_freq//2, 2)):
            neighbours = []
            i = 0
            next_freq = freq
            while next_freq <= max_freq+1:
                neighbours.append(next_freq-1)
                i+=1
                next_freq = freq * (2**i)
            self.harmonic_groups.append(neighbours)