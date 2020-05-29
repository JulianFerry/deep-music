import json
import re
import math
import numpy as np
import librosa
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from IPython.display import Audio, display
from warnings import warn, filterwarnings


class AudioDataset:
    def __init__(self, path='../data/raw/nsynth-train/'):
        """
        Load names and metadata of audio files in the specified directory

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
            self.file_names.setdefault(instrument, []).append(file)
            self.file_names_nested.setdefault(instrument, {}).setdefault(prefix, []).append(file)
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
            raise(ValueError(f'Instrument {instrument} is not one of {self.unique_instruments}'))

    def get_filepath(self, instrument=None, file_index=None, file_name=None):
        """
        Returns the path of an audio file in the dataset.
        There are two ways of loading a file path:
            - Either specify file_name to fetch an NSynth file path by name
            - Alternatively, `instrument` restricts files to those starting with that string
              and `file_index` is used to reference the file index within that subset of files

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
            raise(ValueError('A file_name or both instrument and file_index must be specified'))
        path = self.path/'audio'/file.name
        info = self.examples[file.stem]
        return path, info


class AudioFile:
    """
    Defines methods to load, process and visualise audio waveforms and spectrograms
    Once an AudioFile object is created, methods can also be applied to audio and
    spectrogram objects which are not attributes of the AudioFile.

    Methods:
        __init__ - load audio data from a .wav file
        trim_audio - trim start and end times of the audio waveform
        filter_audio - apply a butterworth filter (lp, hp, bp, bs) to the audio
        plot_audio - plot the audio waveform in the time domain
        play_audio - create a widget to play that audio
        audio_to_spectrogram - generate an FFT spectrogram from the audio waveform
        audio_to_log_spectrogram - generate a CQT spectrogram from the audio waveform
        filter_harmonics - filter out all non-harmonic frequencies from the spectrogram
        spectrogram_to_audio - convert a spectrogram back to an audio waveform
        plot_spectrogram - plot a spectrogram
        convolve_spectrogram (experimental) - apply non-continuous 1-D convolution
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
        Trim the audio waveform to start and end times (in seconds)

        Args:
            start - numeric - start time (seconds)
            end - numeric - end time (seconds)
        """
        start_idx = 0
        if start != 0:
            if 0 < start < self.duration:
                start_idx = int(start * self.sampling_rate)
            else:
                warn('Start set to 0')
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
        Apply a butterworth filter to the audio waveform

        Args:
            lowcut - numeric - lowest frequency
            highcut - numeric - highest frequency
            btype - one of ('lowpass', 'highpass', 'bandpass', 'bandstop')
            order - int - (1, 2, 3, 4, 5) complexity of the butterworth filter
        """
        # Filter parameters
        if btype.startswith('low'):
            Wn = highcut
        elif btype.startswith('high'):
            Wn = lowcut
        elif btype.startswith('band'):
            Wn = np.array([lowcut, highcut])
        else:
            raise(ValueError('Filter should be one of lowpass, highpass, bandpass or bandstop'))
        # Create and apply filter
        b, a = butter(order, Wn / self.nyquist, btype=btype)
        self.audio = lfilter(b, a, self.audio)

    def plot_audio(self, audio=None):
        """
        Plot an audio waveform

        Args:
            audio - np.array - audio to play. None defaults to self.audio
        """
        audio = self.audio if audio is None else audio
        time = np.linspace(0., self.duration, audio.shape[0])
        plt.figure(figsize=(10, 3))
        plt.plot(time, audio)
        plt.xlim(time.min(), time.max())
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

    def play_audio(self, audio=None, sampling_rate=None, autoplay=False):
        """
        Create widget which plays audio
        Defaults to playing the AudioFile's audio if audio is not specified

        Args:
            audio - np.array - audio waveform to play
            sampling_rate - int - audio frequency sampling rate
            autoplay - bool - whether to automatically play the sound from the widget
        """
        audio = self.audio if audio is None else audio
        sampling_rate = self.sampling_rate if sampling_rate is None else sampling_rate
        display(Audio(audio, rate=sampling_rate, autoplay=autoplay))

    # Audio to spectrogram
    def audio_to_spectrogram(self, audio=None, time_intervals=1, resolution=1., mode='max'):
        """
        Convert an audio waveform to a spectrogram using STFT
        Defaults to converting the AudioFile's audio if audio is not specified

        Args:
            time_intervals - int - how many time intervals to split the audio signal into
            resolution - 0 < float <= 1 - the frequency resolution to use
            mode - str - 'max' maximises the frequency resolution
                         'fast' sacrifices some resolution for speed (50% faster)
        """
        audio = self.audio if audio is None else audio
        # setting resolution=1 maximises the frequency domain resolution
        fft_size = self._get_fft_size(resolution, mode)
        self.hop_length = int(self.audio.shape[0] / time_intervals)
        # Create spectrogram
        self.spectrogram = librosa.core.stft(
            self.audio,
            n_fft=fft_size,
            hop_length=self.hop_length + 1,
        )

    def _get_fft_size(self, resolution=1., mode='max'):
        """
        Get fft_size for a given frequency resolution

        resolution - float <= 1 - 1 is maximal frequency resolution
        mode - str - 'max' uses the audio length for fft_size (n_fft in librosa)
                     'fast' uses the largest power of 2 smaller than the audio length
        """
        if resolution > 1:
            raise(ValueError('resolution should be a non-zero value between 0 and 1'))
        if mode == 'max':
            fft_size = int(resolution * self.audio.shape[0] / 2) * 2
        elif mode == 'fast':
            fft_size = 2**int(np.log2(resolution * self.audio.shape[0]))
        return fft_size

    def audio_to_cqt_spectrogram(self, audio=None, time_intervals=1, resolution=1,
                                 min_freq=32.7, max_freq=-1):
        """
        Transform audio waveform data into a CQT spectrogram (log2 frequency resolution)
        Defaults to converting the AudioFile's audio if audio is not specified

        Args:
            audio - np.array - audio waveform data
            time_intervals - int - number of time intervals to split the audio into
            resolution - int - frequency bins per musical note
            min_freq - numeric - minimum frequency for the transform
            max_freq - numeric - maximum frequency for the transform
        """
        # Setup
        audio = self.audio if audio is None else audio
        max_freq = self._get_max_freq(max_freq)
        self._set_cqt_params(audio, time_intervals, resolution, min_freq, max_freq)
        self.spectrogram_cqt = librosa.cqt(
            y=audio,
            sr=self.sampling_rate,
            **self.cqt_params,
            n_bins=self.cqt_bins
        )

    def _set_cqt_params(self, audio, time_intervals, note_resolution, min_freq, max_freq):
        """
        Set the parameters to be be used in CQT transforms, inverse transforms and plots

        Args:
            audio - np.array - audio waveform to convert to CQT
            time_intervals - int - minimum number of time intervals to split the audio
            note_resolution - int - number of frequency bins per musical note
            min_freq - numeric - minimum frequency for the CQT
            max_freq - numeric - maximum frequency for the CQT
        """
        bins_per_octave = int(12 * note_resolution)
        num_octaves = np.log2(max_freq/min_freq)
        int_octaves = int(math.ceil(num_octaves)) - 1
        hop_length = audio.shape[0] / time_intervals
        hop_length = int(hop_length // (2**int_octaves)) * (2**(int_octaves))  # pow 2
        self.cqt_params = {
            'bins_per_octave': bins_per_octave,
            'fmin': min_freq,
            'hop_length': hop_length,
            'scale': False,
            'sparsity': 0,     # high-res but possibly slow
            'res_type': 'fft'  # high-res but possibly slow
        }
        self.cqt_bins = int(num_octaves * bins_per_octave)

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

    # Spectrogram to audio
    def spectrogram_to_audio(self, spectrogram=None, play=True):
        """
        Convert spectrogram to an audio waveform
        Defaults to converting the AudioFile's spectrogram if spectrogram is not specified

        Args:
            spectrogram - np.array - spectrogram to convert to audio
            play - bool - whether to create an Audio widget to play the sound
        """
        spectrogram = self.spectrogram if spectrogram is None else spectrogram
        hop_length = self.hop_length
        if spectrogram.shape[1] < 2:
            spectrogram = np.hstack([spectrogram[:, :1] for _ in range(2)])
            hop_length = int(hop_length * 0.5)
        # Normal FFT
        if np.issubdtype(spectrogram.dtype, np.complexfloating):
            f_inverse = librosa.istft
        else:
            f_inverse = librosa.griffinlim
        recovered_audio = f_inverse(
            spectrogram,
            hop_length=hop_length + 1
        )
        if play:
            self.play_audio(recovered_audio)

    def cqt_spectrogram_to_audio(self, spectrogram=None, play=True):
        """
        Convert a CQT spectrogram to an audio waveform
        Defaults to converting the AudioFile's spectrogram if spectrogram is not specified

        Args:
            spectrogram - np.array - spectrogram to convert to audio
                                     defaults to self.spectrogram_cqt
            play - bool - whether to create an Audio widget to play the sound
        """
        if self.cqt_params is None:
            raise(ValueError('audio_to_cqt_spectrogram has not yet been called'))
        spectrogram = self.spectrogram_cqt if spectrogram is None else spectrogram
        if np.issubdtype(spectrogram.dtype, np.complexfloating):
            f_inverse = librosa.icqt
        else:
            f_inverse = librosa.griffinlim_cqt
            filterwarnings("ignore", category=UserWarning)
        recovered_audio = f_inverse(
            spectrogram,
            sr=self.sampling_rate,
            **self.cqt_params
        )
        filterwarnings("default", category=UserWarning)
        if play:
            self.play_audio(recovered_audio)

    # Spectrogram methods
    def filter_harmonics(self, neighbour_radius=0):
        """
        Remove non-harmonic frequency amplitudes from the audio spectrogram

        Args:
            neighbour_radius - int -  number of neighbouring frequencies to include
        """
        self.spectrogram_harm = np.zeros(self.spectrogram.shape, dtype='complex64')
        step = int(self.fundamental_freq * self.spectrogram.shape[0] / self.nyquist)
        for t in range(self.spectrogram.shape[1]):
            for i in range(neighbour_radius+1):
                self.spectrogram_harm[step+i::step, t] = self.spectrogram[step+i::step, t]
                self.spectrogram_harm[step-i::step, t] = self.spectrogram[step-i::step, t]

    def plot_spectrogram(self, spectrogram=None, spec_thresh=0, min_freq=None, max_freq=None,
                         y_harm=None, cqt=False, plot_phase=False, title='', figsize=(10, 6),
                         **kwargs):
        """
        Plot a spectrogram as a matplotlib color mesh
        Defaults to plotting the AudioFile's spectrogram if spectrogram is not specified

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
        # Params
        spectrogram = self.spectrogram if spectrogram is None else spectrogram
        if cqt and (self.cqt_params is None):
            raise(ValueError('audio_to_cqt_spectrogram has not yet been called'))
        max_freq = self._get_max_freq(max_freq)
        min_freq = self._get_min_freq(min_freq, cqt)
        plt.figure(figsize=figsize)
        plt.subplot(1+plot_phase, 1, 1)
        if cqt:
            kwargs['bins_per_octave'] = self.cqt_params['bins_per_octave']
            kwargs['fmin'] = self.cqt_params['fmin']
            kwargs['hop_length'] = self.cqt_params['hop_length']
        # Plot
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(spectrogram)).clip(min=spec_thresh),
            sr=self.sampling_rate,
            cmap=plt.cm.afmhot,
            y_axis='cqt_hz' if cqt else 'hz',
            **kwargs
        )
        plt.ylim(min_freq, max_freq)
        plt.title(title)
        plt.colorbar(format='%+2.0f dB')
        # Change yticks
        if y_harm is not None:
            ytick_freqs = self._harmonic_ticks(min_freq, max_freq, cqt, y_harm)
            plt.yticks(ytick_freqs)
        plt.show()

    def plot_fft(self, fft=None, min_freq=None, max_freq=-1, x_harm=None, cqt=False,
                 title='', figsize=(15, 4), **kwargs):
        """
        Plot a fft as a matplotlib line plot
        Defaults to plotting the first time bin of the AudioFile's spectrogram
        if fft is not specified

        Args:
            spectrogram - np.array - spectrogram to plot
            min_freq - numeric - plot minimum frequency
            max_freq - numeric - plot maximum frequency - 'default' uses 10x the fundamental
            x_harm - whether to use harmonic frequencies as y axis labels
            cqt - bool - whether the spectrogram frequencies are on a cqt scale
            title - str - plot title
            figsize - tuple - (width, height)
            ax - matplotlib ax
        """
        # Params
        fft = self.spectrogram[:, 0] if fft is None else fft
        if cqt and (self.cqt_params is None):
            raise(ValueError('audio_to_cqt_spectrogram has not yet been called'))
        max_freq = self._get_max_freq(max_freq)
        min_freq = self._get_min_freq(min_freq, cqt)
        plt.figure(figsize=figsize)
        if cqt:
            freqs = librosa.core.cqt_frequencies(
                fft.shape[0], fmin=self.cqt_params['fmin'],
                bins_per_octave=self.cqt_params['bins_per_octave'])
        else:
            freqs = np.linspace(0, self.nyquist, len(fft))
        # Plot
        plt.plot(
            freqs,
            10 * np.log10(np.abs(fft)),
            **kwargs
        )
        plt.xlim(min_freq, max_freq)
        plt.title(title)
        # Change xticks
        if cqt:
            plt.xscale('log', basex=2)
            axis = plt.gca().xaxis
            axis.set_major_formatter(librosa.display.LogHzFormatter())
            axis.set_major_locator(librosa.display.LogLocator(base=2.0))
            axis.set_minor_formatter(librosa.display.LogHzFormatter(major=False))
        if x_harm is not None:
            xtick_freqs = self._harmonic_ticks(min_freq, max_freq, cqt, x_harm)
            plt.xticks(xtick_freqs)
        plt.show()

    def _harmonic_ticks(self, min_freq, max_freq, cqt, num_harm):
        """
        Set the harmonic frequencies as axis labels when plotting in the frequency domain

        Args:
            min_freq - numeric - minimum frequency of the plot
            max_freq - numeric - maximum frequency of the plot
            cqt - bool - whether the plot is a cqt plot (log2 frequency domain)
            num_harm - int - number of harmonics to plot (for cqt plot only)
        """
        axis_min = self.fundamental_freq * ((min_freq // self.fundamental_freq)+1)
        axis_max = self.fundamental_freq * (max_freq // self.fundamental_freq)
        if cqt:
            axis_max = min(axis_max, self.fundamental_freq * num_harm)
            axis_step = self.fundamental_freq
        else:
            axis_step = (axis_max - axis_min) / 9
        return range(int(axis_min), int(axis_max)+1, int(axis_step))

    # Convolutions
    def convolve_spectrogram(self, spectrogram):
        """
        Unravel the spectrogram, apply 1-D conv to it then 'ravel' it back
        """
        self._get_harmonic_groups(spectrogram)
        self.spectrogram_conv = np.ones(spectrogram.shape)[::2]
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
