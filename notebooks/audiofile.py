import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
from IPython.display import Audio, display
from warnings import warn
import math

from mel import butterworth_filter, pretty_spectrogram, invert_pretty_spectrogram
from mel import create_mel_filter, make_mel, mel_to_spectrogram


class AudioFile:
    """
    Defines methods to load, process and visualise audio data using spectrograms
    
    Methods:
        __init__ - fetch the path of an audio file in '../data/raw/nsynth-*'
        load_audio - load data from the .wav audio file
        plot_audio - plot the audio waveform in the time domain
        play_audio - create a widget to play that audio
        audio_to_spectrogram - generate an FFT spectrogram from the audio data
        spectrogram_to_audio - convert a spectrogram back to an audio waveform
        plot_spectrogram - plot a spectrogram
        filter_harmonics - filter out all non-harmonic frequencies from the spectrogram
        convolve_spectrogram (experimental) - apply non-continuous 1-D convolution
    """
    
    def __init__(self, dataset, instrument, file_index):
        """
        Fetch the path of an audio file in '../data/raw/nsynth-*'
        The files will be restricted to those starting with the `instrument` string
        `file_index` is used to select the file within that subset of files
        
        Args:
            dataset - str - Nsynth dataset folder ('train', 'valid', 'test')
            instrument - str - ('keyboard_acoustic', 'guitar_acoustic')
            file_index - int - index of the file within the specified dataset 
        """
        # Fetch audio file path
        data_dir = Path(f'../data/raw/nsynth-{dataset}/')
        examples = json.load(open(data_dir/'examples.json'))
        file_names = pd.Series(list(examples.keys()))
        file = Path(
            file_names[file_names.str.startswith(instrument)].iloc[file_index] + '.wav'
        )
        self.path = data_dir/'audio'/file.name
        self.info = examples[file.stem]
        self.fundamental_freq = None
        # Store unique instruments found in that dataset
        re_find_instruments = partial(re.findall, '^.*?\d')
        self.unique_instruments = (file_names
                                   .apply(re_find_instruments)
                                   .str[0]
                                   .unique())
        
    # Audio
    def load_audio(self, start=0, end=-1, butter=None):
        """
        Load audio data from .wav file identified in __init__()
        
        Args:
            start - int - when to start the audio clip, in seconds
            end - int - when to end the audio clip, in seconds
            butter - tuple - butterworth filter (low, high, btype)
        """
        self.sampling_rate, self.audio = wavfile.read(self.path)
        self.nyquist = self.sampling_rate // 2
        duration = self.audio.shape[0] / self.sampling_rate  # in seconds
        # Clip audio duration
        if end != -1:
            if start < end <= duration:
                end = int(end * self.sampling_rate)
            else:
                warn(f'End set to {duration}')
                end = -1
        if start != 0:
            if 0 < start < duration:
                start = int(start * self.sampling_rate)
            else:
                warn(f'Start set to 0')
                start = 0
        self.audio = self.audio[start:end]
        self.duration = self.audio.shape[0] / self.sampling_rate
        # Butterworth filter
        if butter:
            self.audio = butterworth_filter(self.audio, self.sampling_rate, *butter, order=4)
    
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

    def play_audio(self):
        """
        Create widget which plays audio
        """
        display(Audio(self.audio, rate=self.sampling_rate))
    
    # Spectrograms
    def audio_to_spectrogram(self, time_intervals=1, spec_thresh=4):
        """
        Stolen from: https://timsainburg.com/python-mel-compression-inversion.html

        Args:
            time_intervals: how many time intervals to create a spectrogram for   
            spec_thresh: threshold for spectrograms (lower filters out more noise)
        """
        # FFT params - fft_size is maximised to maximise the frequency domain resolution
        self.time_intervals = time_intervals
        fft_size = (self.audio.shape[0] // 2) * 2
        step_size = fft_size // self.time_intervals
        # Create spectrogram
        self.spectrogram = pretty_spectrogram(
            self.audio.astype("float64"),
            fft_size=fft_size,
            step_size=step_size,
            log=True,
            thresh=spec_thresh,
        )

    def spectrogram_to_audio(self, spectrogram, time_intervals=1):
        """
        Convert spectrogram to audio
        """
        fft_size = spectrogram.shape[1] * 2
        step_size = fft_size // self.time_intervals
        recovered_audio = invert_pretty_spectrogram(
            spectrogram,
            fft_size=fft_size,
            step_size=step_size,
            log=True,
            n_iter=10
        )
        display(Audio(recovered_audio, rate=self.sampling_rate))
        
    def _get_plot_ylim(self, spectrogram, min_freq, max_freq, log_scale):
        """
        Calculate a spectrogram's bounding indices from bounding frequencies
        
        Args:
            spectrogram - np.array - spectrogram for which to calculate indices
            min_freq - float - minimum frequency to find index for
            max_freq - float - maximum frequency to find index for
            log_scale - bool - whether or not the frequencies are on a log scale
        """
        # ymin
        if min_freq <= 1:
            ymin = 0
        else:
            ymin = self._log_to_idx(self._freq_to_log(min_freq))
        # ymax
        if max_freq == 'default':
            if self.fundamental_freq:
                max_freq = self.fundamental_freq * 10
                if log_scale:
                    ymax = self._log_to_idx(self._freq_to_log(max_freq))
                else:
                    ymax = spectrogram.shape[1] * max_freq // self.nyquist
            else:
                max_freq = self.nyquist
                ymax = -1
        elif max_freq == -1:
            max_freq = self.nyquist
            ymax = -1
        else:
            if log_scale:
                ymax = self._log_to_idx(self._freq_to_log(max_freq))
            else:
                ymax = spectrogram.shape[1] * max_freq // self.nyquist
        return min_freq, max_freq, ymin, ymax
    
    def plot_spectrogram(self, spectrogram, title='', figsize=(10, 6), log_scale=False,
                         min_freq=0, max_freq='default'):
        """
        Plot a spectrogram
        
        Args:
            spectrogram - np.array - spectrogram to plot
            title - str - plot title
            figsize - tuple - (width, height)
            log_scale - bool - whether to label the y axis on a log scale
            min_freq - numeric - plot minimum frequency
            max_freq - numeric - plot maximum frequency - 'default' uses 10x the fundamental
        """
        min_freq, max_freq, ymin, ymax = \
            self._get_plot_ylim(spectrogram, min_freq, max_freq, log_scale)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        cax = ax.matshow(
            spectrogram.T[ymin:ymax],
            interpolation="nearest",
            aspect="auto",
            cmap=plt.cm.afmhot,
            origin="lower",
        )
        ax.get_xaxis().set_visible(False)
        if log_scale:
            ytick_freqs = range(
                (ymin//self.fundamental_freq + 1) * self.fundamental_freq,
                max_freq+1,
                self.fundamental_freq
            )
            plt.yticks([self._log_to_idx(self._freq_to_log(x))-ymin for x in ytick_freqs])
            ax.set_yticklabels(ytick_freqs)
        fig.colorbar(cax)
        plt.title(title)
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

    def _idx_to_freq(self, idx):
        return idx*self.hz_per_idx
    def _freq_to_log(self, freq):
        return np.log2(freq)
    def _log_to_idx(self, log_freq):
        return int(log_freq * 10000)
    def _idx_to_log_idx(self, idx):
        return self._log_to_idx(self._freq_to_log(self._idx_to_freq(idx)))
    def spectrogram_to_log(self, spectrogram):
        """ Convert a spectrogram to the log2 domain
        
        Args:
            spectrogram - spectrogram to convert to log domain
        """
        # Setup
        self.hz_per_idx = self.nyquist / spectrogram.shape[1]
        idx_one_hz = math.ceil(1 / self.hz_per_idx)
        max_log_idx = self._idx_to_log_idx(spectrogram.shape[1])
        self.spectrogram_log = np.zeros((spectrogram.shape[0], max_log_idx))
        # Calculate log spectrogram
        for t in range(spectrogram.shape[0]):
            prev_log_idx = 0
            for idx in range(idx_one_hz, spectrogram.shape[1]):
                log_idx = self._idx_to_log_idx(idx)
                self.spectrogram_log[t, prev_log_idx:log_idx] = spectrogram[t, idx]
                prev_log_idx = log_idx        
        
    def filter_harmonics(self, neighbour_radius=0):
        """
        Remove all frequency amplitudes and only keep harmonics of the fundamental frequency
        
        Args:
            neighbour_radius - int -  number of neighbouring frequencies to include
        """
        self.spectrogram_harm = np.ones(self.spectrogram.shape) * self.spectrogram.min()
        step = self.fundamental_freq * self.spectrogram.shape[1] // self.nyquist
        for i in range(neighbour_radius+1):
            self.spectrogram_harm[0, step+i::step] = self.spectrogram[0, step+i::step]
            self.spectrogram_harm[0, step-i::step] = self.spectrogram[0, step-i::step]
    
    # Convolutions
    def _get_harmonic_groups(self):
        """
        Create groups of harmonic frequencies to unwrap the spectrogram
        """
        max_freq = self.spectrogram.shape[1]
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

    def convolve_spectrogram(self):
        """
        Unravel the spectrogram, apply 1-D conv to it then 'ravel' it back
        """
        self._get_harmonic_groups()
        self.spectrogram_conv = np.ones(self.spectrogram.shape)[:, ::2] # * -8
        mask = [1, 1]
        for t in range(self.spectrogram.shape[0]):
            # Time slice to apply convolution to
            spectrogram_slice = self.spectrogram[t]
            # Convolve each group of harmonics
            harmonic_convs = []
            for group in self.harmonic_groups:
                harmonic_amplitudes = spectrogram_slice[group]
                harmonic_convs = np.convolve(mask, harmonic_amplitudes, mode='valid')
                self.spectrogram_conv[t, group[:-1]] = harmonic_convs