import math
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from IPython import display
from scipy.signal import butter, lfilter

# Fix circular imports with absolute import
from importlib import import_module
pkg = __name__.split('.')[0]
spec_ = import_module(f'{pkg}.spectrogram')


class Audio(np.ndarray):
    """
    Wrapper around numpy array, intended to handle audio waveform data.
    Contains attributes specific to audio such as sampling rate and fundamental frequency,
    and defines methods to display and process audio and convert it to spectrograms.

    Methods
    -------
    plot:
        Plot the audio waveform in the time domain
    play:
        Create a widget to play the audio waveform
    trim:
        Trim start and end times of the audio waveform
    filter:
        Apply a butterworth filter (lp, hp, bp, bs) to the audio
    to_spectrogram:
        Generate a spectrogram (STFT or CQT) from the audio

    """

    # Numpy methods

    def __new__(
        cls,
        array: np.array,
        sampling_rate: int,
        fundamental_freq: float = None
    ):
        """
        Cast a numpy array to an Audio object and set the __dict__ attributes.

        Parameters
        ----------
        array: np.array
            Audio data to convert to Audio object
        sampling_rate: int
            Audio sampling rate
        fundamental_freq: int or float
             Fundamental frequency of the audio (if known)

        """
        obj = np.asarray(array).astype(np.float).view(cls)
        obj.sampling_rate = sampling_rate
        obj.nyquist = sampling_rate / 2
        obj.duration = len(array) / sampling_rate
        obj.fundamental_freq = fundamental_freq
        return obj

    def __array_finalize__(self, obj):
        """
        Numpy subclassing constructor. This gets called every time an Audio
        object is created, either by using the Audio() constructor or when
        an Audio method returns self.
        See https://numpy.org/devdocs/user/basics.subclassing.html

        """
        if obj is None: return  # noqa
        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        self.fundamental_freq = getattr(obj, 'fundamental_freq', None)
        if type(obj) is type(self):
            self.nyquist = self.sampling_rate / 2
            self.duration = self.shape[0] / self.sampling_rate
        else:
            self.nyquist, self.duration = None, None

    # Audio methods

    def play(
        self,
        autoplay: bool = False
    ):
        """
        Create widget which plays audio.

        Parameters
        ----------
        autoplay: bool
            Whether to automatically play sound from the widget

        """
        display.display(display.Audio(self, rate=self.sampling_rate, autoplay=autoplay))

    def plot(self):
        """
        Plot the audio waveform

        """
        time = np.linspace(0., self.duration, self.shape[0])
        plt.figure(figsize=(10, 3))
        plt.plot(time, self)
        plt.xlim(time.min(), time.max())
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

    def trim(
        self,
        start: float = 0,
        end: float = -1
    ):
        """
        Trim the audio start and end times (in seconds).

        Parameters
        ----------
        start: int or float
            Start time (seconds)
        end: int or float
            End time (seconds)

        Returns
        -------
        trimmed_audio: audiolib.Audio
            Modified version of the Audio object

        """
        if 0 < start < self.duration:
            start_idx = int(start * self.sampling_rate)
        else:
            start_idx = 0
            if start != 0:
                warn('Start set to 0')
        if start < end <= self.duration:
            end_idx = int(end * self.sampling_rate)
        else:
            end_idx = -1
            if end != -1:
                warn(f'End set to {self.duration}')
        return self[start_idx:end_idx]

    def filter(
        self,
        lowcut: float = None,
        highcut: float = None,
        btype: str = 'lowpass',
        order: int = 4
    ):
        """
        Apply a butterworth filter to the audio waveform.

        Parameters
        ----------
        lowcut: int or float
            Lowest frequency
        highcut: int or float
            Highest frequency
        btype: ('lowpass', 'highpass', 'bandpass', 'bandstop')
            Butterworth filter type
        order: (1, 2, 3, 4, 5)
            Complexity of the butterworth filter

        Returns
        -------
        filtered_audio: audiolib.Audio
            Modified version of the Audio object

        """
        # Filter parameters
        if btype.startswith('low'):
            if highcut is None:
                raise(TypeError('highcut needs to be specified for a lowpass filter'))
            Wn = highcut
        elif btype.startswith('high'):
            if lowcut is None:
                raise(TypeError('lowcut needs to be specified for a highpass filter'))
            Wn = lowcut
        elif btype.startswith('band'):
            if (lowcut is None) and (highcut is None):
                raise(TypeError(
                    f'lowcut and highcut need to be specified for a {btype} filter'))
            Wn = np.array([lowcut, highcut])
        else:
            raise(ValueError(
                'Filter btype should be one of: lowpass, highpass, bandpass or bandstop'))
        # Create and apply filter
        b, a = butter(order, Wn / self.nyquist, btype=btype)
        filtered_audio = lfilter(b, a, self)
        filtered_audio = Audio(
            filtered_audio,
            self.sampling_rate,
            self.fundamental_freq
        )
        return filtered_audio

    def to_spectrogram(
        self,
        time_intervals: int = 1,
        resolution: int = 1,
        mode: str = 'max',
        cqt: bool = False,
        fmin: float = 32.7,
        fmax: float = -1
    ):
        """
        Convert the audio waveform to a spectrogram using STFT or CQT.

        Parameters
        ----------
        time_intervals: int
            Number of time intervals to split the audio signal into
        resolution: int or float
            Frequency resolution
            - if `cqt=False`: 0 < float <= 1 - proportional to n_fft
            - if `cqt=True`:  0 < int        - bins per musical note
        mode: ('fast', 'max')
            'fast' sacrifices some resolution for speed (30% faster)
            'max' maximises the frequency resolution (`cqt=False` only)
        fmin: int or float
            minimum frequency for the transform (`cqt=True` only)
        fmax: int or float
            maximum frequency for the transform (`cqt=True` only)

        Returns
        -------
        spectrogram: audiolib.Spectrogram
            Spectrogram with the parameters used to generate it as attributes

        """
        # CQT
        if cqt is True:
            return self._to_cqt_spectrogram(time_intervals, resolution, fmin, fmax)
        # STFT
        n_fft = self._get_n_fft(resolution, mode)
        hop_length = int(self.shape[0] / time_intervals) + 1
        spectrogram = librosa.core.stft(
            np.asarray(self),
            n_fft=n_fft,
            hop_length=hop_length,
        )
        spectrogram = spec_.Spectrogram(
            spectrogram,
            self.sampling_rate,
            self.fundamental_freq,
            params={'hop_length': hop_length}
        )
        return spectrogram

    def _to_cqt_spectrogram(
        self,
        time_intervals: int = 1,
        resolution: int = 1,
        fmin: float = 32.7,
        fmax: float = -1
    ):
        """
        Transform audio waveform data into a CQT spectrogram (log2 frequency resolution)

        Parameters
        ----------
        time_intervals: int
            Number of time intervals to split the audio into
        resolution: int
            Number of frequency bins per musical note
        fmin: int or float
            Minimum frequency for the transform
        fmax: int or float
            Maximum frequency for the transform

        Returns
        -------
        cqt_spectrogram: Spectrogram
            CQT Spectrogram with the parameters used to generate it as attributes

        """
        # Setup
        fmax = self._get_fmax(fmax)
        cqt_params = self._get_cqt_params(time_intervals, resolution, fmin, fmax)
        spectrogram = librosa.cqt(
            y=np.asarray(self),
            sr=self.sampling_rate,
            **cqt_params,
        )
        spectrogram = spec_.Spectrogram(
            spectrogram,
            self.sampling_rate,
            self.fundamental_freq,
            cqt=True,
            params=cqt_params
        )
        return spectrogram[:, :-1]

    def _get_n_fft(
        self,
        resolution: float = 1.,
        mode: str = 'max'
    ):
        """
        Get n_fft for a given frequency resolution between 0 and 1.

        Parameters
        ----------
        resolution: float
            0 < r <= 1  (1 maximises frequency resolution)
        mode: ('max', 'fast')
            'max' uses the number of audio samples as n_fft
            'fast' uses the largest power of 2 smaller than the number of samples

        """
        if resolution > 1:
            raise(ValueError(
                'For `cqt=False`, resolution should be a non-zero value between 0 and 1'))
        if mode == 'max':
            n_fft = int(resolution * self.shape[0] / 2) * 2
        elif mode == 'fast':
            n_fft = 2**int(np.log2(resolution * self.shape[0]))
        return n_fft

    def _get_fmax(
        self,
        fmax: float
    ):
        """
        Calculate the default max frequency for plotting, based on the audio nyquist
        and fundamental frequency.

        """
        if fmax is None:
            fmax = 10 * self.fundamental_freq
        elif fmax == -1:
            fmax = self.nyquist
        return fmax

    def _get_cqt_params(
        self,
        time_intervals: int,
        note_resolution: int,
        fmin: float,
        fmax: float
    ):
        """
        Set the parameters to be be used the CQT transform.

        Parameters
        ----------
        time_intervals: int
            Minimum number of time intervals to split the audio
        note_resolution: int
            Number of frequency bins per musical note
        fmin: float
            Minimum frequency for the CQT
        fmax: float
            Maximum frequency for the CQT

        """
        bins_per_octave = int(12 * note_resolution)
        num_octaves = np.log2(fmax/fmin)
        int_octaves = int(math.ceil(num_octaves)) - 1
        hop_length = self.shape[0] / time_intervals
        hop_length = int(hop_length // (2**int_octaves)) * (2**(int_octaves))  # pow 2
        cqt_params = {
            'n_bins': int(num_octaves * bins_per_octave),
            'bins_per_octave': bins_per_octave,
            'hop_length': hop_length,
            'fmin': fmin,
            'scale': False,
            'sparsity': 0,     # high-res but possibly slow
            'res_type': 'fft'  # high-res but possibly slow
        }
        return cqt_params
