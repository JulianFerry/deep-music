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

from fft import butterworth_filter, pretty_spectrogram, invert_pretty_spectrogram
from fft import create_mel_filter, make_mel, mel_to_spectrogram


class AudioDataset:
    def __init__(self, path='../data/raw/nsynth-train/', metadata='examples.json'):
        """
        Load names and metadata of audio files in the specified directory
        
        Args:
            dataset - str - Nsynth dataset folder ('train', 'valid', 'test')
            instrument - str - ('keyboard_acoustic', 'guitar_acoustic')
            file_index - int - index of the file within the specified dataset 
        """
        if type(path) is str:
            self.path = Path(path)
        elif type(path) is Path:
            self.path = path
        else:
            raise(ValueError('path argument should be {} or {}'.format(str, Path)))
        self.abspath = self.path.resolve()
        # Load metadata
        self.examples = json.load(open(self.abspath/'examples.json'))
        self.file_names = pd.Series(list(self.examples.keys()))
        re_find_instruments = partial(re.findall, '(^.*?)_\d')
        self.unique_instruments = (self.file_names
                                   .apply(re_find_instruments)
                                   .str[0]
                                   .unique())
        
    def get_filepath(self, instrument, file_index, folder='audio', ext='.wav'):
        """
        Returns the path of an audio file in the dataset. Files will be restricted
        to those starting with the `instrument` string - `file_index` is used to
        select the file within that subset of files.
        
        Args:
            instrument - str - one of ('keyboard_acoustic', 'guitar_acoustic')
            file_index - int - index of the file within the instrument's subset of files
        """
        file = Path(
            self.file_names[self.file_names.str.startswith(instrument)].iloc[file_index] + ext
        )
        path = self.abspath/'audio'/file.name
        info = self.examples[file.stem]        
        return path, info


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
    
    def __init__(self, path, info=None):
        """
        Load audio data from .wav file
        
        Args:
            start - int - when to start the audio clip, in seconds
            end - int - when to end the audio clip, in seconds
            butter - tuple - butterworth filter (low, high, btype)
        """
        self.path = path
        self.info = info
        self.fundamental_freq = None
        self.log_resolution = None
        self._load_audio()
        
    # Audio methods
    def _load_audio(self):
        """
        Load audio from the path specified in __init__
        Automatically called on class creation
        """
        self.sampling_rate, self.audio = wavfile.read(self.path)
        self.nyquist = self.sampling_rate // 2
        self.duration = self.audio.shape[0] / self.sampling_rate

    def trim_audio(self, start=0, end=-1, reload=True):
        """
        Trim the audio data to start and end times (in seconds)
        
        Args:
            start - numeric - start time (seconds)
            end - numeric - end time (seconds)
        """
        if reload:
            self._load_audio()
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

    def play_audio(self):
        """
        Create widget which plays audio
        """
        display(Audio(self.audio, rate=self.sampling_rate))
    
    # Spectrogram methods
    def audio_to_spectrogram(self, time_intervals=1, spec_thresh=4):
        """
        Stolen from: https://timsainburg.com/python-mel-compression-inversion.html

        Args:
            time_intervals: how many time intervals to create a spectrogram for   
            spec_thresh: threshold for spectrograms (lower filters out more noise)
        """
        # fft_size is maximised to maximise the frequency domain resolution
        fft_size = (self.audio.shape[0] // 2) * 2
        self.time_intervals = time_intervals
        step_size = fft_size // self.time_intervals
        # Create spectrogram
        self.spectrogram = pretty_spectrogram(
            self.audio.astype("float64"),
            fft_size=fft_size,
            step_size=step_size,
            log=True,
            thresh=spec_thresh,
        )
        self.hz_per_idx = self.nyquist / self.spectrogram.shape[1]

    def spectrogram_to_audio(self, spectrogram):
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

    def filter_harmonics(self, neighbour_radius=0):
        """
        Only keep spectrogram amplitudes for harmonics of the fundamental frequency
        
        Args:
            neighbour_radius - int -  number of neighbouring frequencies to include
        """
        self.spectrogram_harm = np.ones(self.spectrogram.shape) * self.spectrogram.min()
        step = self.fundamental_freq * self.spectrogram.shape[1] // self.nyquist
        for i in range(neighbour_radius+1):
            self.spectrogram_harm[0, step+i::step] = self.spectrogram[0, step+i::step]
            self.spectrogram_harm[0, step-i::step] = self.spectrogram[0, step-i::step]
    
    def spectrogram_to_log(self, spectrogram, **kwargs):
        """ Convert a spectrogram to the log2 domain
        
        Args:
            spectrogram - spectrogram to convert to log domain
        """
        # Setup
        if not self.log_resolution:
            resolution = self.calculate_log_resolution(spectrogram)
            self.set_log_resolution(resolution)
        idx_one_hz = math.ceil(1 / self.hz_per_idx)
        max_log_idx = self._idx_to_logidx(spectrogram.shape[1], **kwargs)
        self.spectrogram_log = np.zeros((spectrogram.shape[0], max_log_idx))
        # Convert spectrogram to log
        for t in range(spectrogram.shape[0]):
            prev_log_idx = 0
            for idx in range(idx_one_hz, spectrogram.shape[1]):
                log_idx = self._idx_to_logidx(idx, **kwargs)
                self.spectrogram_log[t, prev_log_idx:log_idx] = spectrogram[t, idx]
                prev_log_idx = log_idx

    def set_log_resolution(self, resolution):
        """
        Set the multiplier used to convert log2 frequencies to a `spectrogram_log` array index
        Used by _log_to_logidx()
        
        Args:
            resolution - int - multiplier used to convert log2 frequencies to an array index
        """
        self.log_resolution = resolution

    def calculate_log_resolution(self, spectrogram):
        """
        Calculate the lowest multiplier used to convert log2 frequencies to an array index,
        for which none of the freq->log(freq) mappings overlap in the log domain.
        This ensures that no information is lost when mapping to log frequencies.
        """
        res = 0
        top_n = 10 ** np.arange(1, math.ceil(np.log10(spectrogram.shape[1])))
        top_n = np.append(top_n, spectrogram.shape[1]-1)
        for n in top_n:
            print(f'Finding the lowest resolution multiplier for which the top {n} '
                  'log frequencies do not overlap...')
            logs_overlap = True
            while logs_overlap:
                res+=1
                logs_overlap = False
                for i in range(0, n):
                    logs_overlap = logs_overlap or (
                        self._idx_to_logidx(self.spectrogram.shape[1]-i, resolution=res) == \
                        self._idx_to_logidx(self.spectrogram.shape[1]-i-1, resolution=res))
            print('Resolution multiplier found:', res)
        return res

    def _idx_to_freq(self, idx):
        return idx*self.hz_per_idx
    def _freq_to_idx(self, freq):
        return int(freq/self.hz_per_idx)
    def _freq_to_log(self, freq):
        return np.log2(freq)
    def _log_to_logidx(self, log_freq, **kwargs):
        resolution = kwargs.get('resolution', self.log_resolution)
        return int(log_freq * resolution)
    def _idx_to_logidx(self, idx, **kwargs):
        return self._log_to_logidx(self._freq_to_log(self._idx_to_freq(idx)), **kwargs) 

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
            origin="lower"
        )
        ax.get_xaxis().set_visible(False)
        # Change yticks
        ytick_freqs = range((ymin//self.fundamental_freq + 1) * self.fundamental_freq,
                            max_freq+1,
                            self.fundamental_freq)
        if log_scale:
            plt.yticks([self._log_to_logidx(self._freq_to_log(x)) - ymin for x in ytick_freqs])
        else:
            plt.yticks([self._freq_to_idx(x) - ymin for x in ytick_freqs])
        ax.set_yticklabels(ytick_freqs)
        fig.colorbar(cax)
        plt.title(title)
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

    def _get_plot_ylim(self, spectrogram, min_freq, max_freq, log_scale):
        """
        Calculate a spectrogram's bounding indices from bounding frequencies
        
        Args:
            spectrogram - np.array - spectrogram for which to calculate indices
            min_freq - float - minimum frequency to find index for
            max_freq - float - maximum frequency to find index for
            log_scale - bool - whether or not the spectrogram frequencies are on a log scale
        """
        if log_scale:
            if min_freq <= 1:
                min_freq = 0
                ymin = 0
            else:
                ymin = self._log_to_logidx(self._freq_to_log(min_freq))
            if max_freq == 'default' or max_freq == -1:
                max_freq = self.nyquist
                ymax = -1
            else:
                ymax = self._log_to_logidx(self._freq_to_log(max_freq))
        else:
            ymin = spectrogram.shape[1] * min_freq // self.nyquist
            if max_freq == 'default':
                max_freq = 10 * self.fundamental_freq
                ymax = spectrogram.shape[1] * max_freq // self.nyquist
            elif max_freq == -1:
                max_freq = self.nyquist
                ymax = -1
            else:
                ymax = spectrogram.shape[1] * max_freq // self.nyquist
        return min_freq, max_freq, ymin, ymax
    
    # Convolutions
    def convolve_spectrogram(self, spectrogram):
        """
        Unravel the spectrogram, apply 1-D conv to it then 'ravel' it back
        """
        self._get_harmonic_groups(spectrogram)
        self.spectrogram_conv = np.ones(spectrogram.shape)[:, ::2] # * -8
        mask = [1, 1]
        for t in range(spectrogram.shape[0]):
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
        max_freq = spectrogram.shape[1]
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