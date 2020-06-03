import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from warnings import warn, filterwarnings

# Fix circular imports with absolute import
from importlib import import_module
pkg = __name__.split('.')[0]
audio_ = import_module(f'{pkg}.audio')


class Spectrogram(np.ndarray):
    """
    Wrapper around numpy array, intended to handle audio spectrogram data.
    Contains attributes specific to audio such as sampling rate and fundamental frequency,
    as well as spectrogram specific data such as fft size, hop length and resolution,
    and defines methods to display and process spectrograms and convert them to audio.

    Methods:
    --------
    plot:
        Plot the spectrogram as an image
    plot_fft:
        Plot a time bin of the spectrogram as an FFT line plot
    filter_harmonics:
        Filter out all non-harmonic frequencies from the spectrogram
    to_audio:
        Convert a spectrogram back to an audio waveform
    convolve_spectrogram: (experimental)
        Apply non-continuous 1-D convolution
    """

    # Numpy methods

    def __new__(
        cls,
        array: np.array,
        sampling_rate: int,
        fundamental_freq: int = None,
        cqt: bool = False,
        params: dict = None
    ):
        """
        Cast numpy array to Spectrogram object and set object __dict__ attributes

        Args:
        -----
        sampling_rate: int
            Sampling rate of the original audio data
        fundamental_freq: int or float
            Audio fundamental frequency in Hz
        cqt: bool
            If the spectrogram is a CQT with frequencies on the log2 scale
        params: dict
            Parameters used to create the spectrogram (hop_length or cqt_params)
        """
        obj = np.asarray(array).view(cls)
        obj.sampling_rate = sampling_rate
        obj.nyquist = sampling_rate / 2
        obj.fundamental_freq = fundamental_freq
        obj.cqt = cqt
        obj.params = {} if params is None else params
        return obj

    def __array_finalize__(self, obj):
        """
        Numpy subclassing constructor. This gets called every time a Spectrogram
        object is created, either by using the Spectrogram() constructor or when
        a method returns self.
        See https://numpy.org/devdocs/user/basics.subclassing.html
        """
        if obj is None: return  # noqa
        self.sampling_rate = getattr(obj, 'sampling_rate', None)
        self.fundamental_freq = getattr(obj, 'fundamental_freq', None)
        self.hop_length = getattr(obj, 'hop_length', None)
        self.cqt = getattr(obj, 'cqt', None)
        self.params = getattr(obj, 'params', {})
        if type(obj) is type(self):
            self.nyquist = self.sampling_rate / 2
        else:
            self.nyquist = None

    def __reduce__(self):
        """
        Add __dict__ attributes to the pickled object for `pickle.dump` calls
        """
        pickled_state = super().__reduce__()
        attrs = (self.sampling_rate, self.fundamental_freq, self.cqt, self.params)
        new_state = pickled_state[2] + attrs
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """
        Load __dict__ attributes from pickled objects for `pickle.load` calls
        """
        self.sampling_rate, self.fundamental_freq, self.cqt, self.params = state[-4:]
        self.nyquist = self.sampling_rate / 2
        super().__setstate__(state[0:-4])

    # Spectrogram methods

    def plot(
        self,
        db_thresh: int = 0,
        fmin: float = None,
        fmax: float = None,
        axis_harm: int = None,
        title: str = None,
        figsize: tuple = (10, 6),
        **kwargs
    ):
        """
        Plot a spectrogram as a matplotlib color mesh

        Args:
        -----
        db_thresh: int
            Minimum spectrogram decibel amplitude to plot
        fmin: float
            Minimum frequency to plot
        fmax: float
            Maximum frequency to plot (defaults 10x the fundamental)
        axis_harm: bool
            Whether to use harmonic frequencies as y axis labels
        title: str
            Plot title
        figsize: tuple (width: int, height: int)
            Plot dimensions
        **kwargs:
            Matplotlib plot kwargs (e.g ax)
        """
        # Params
        fmin = self._get_fmin(fmin)
        fmax = self._get_fmax(fmax)
        if self.cqt:
            kwargs['hop_length'] = self.params['hop_length']
            kwargs['bins_per_octave'] = self.params['bins_per_octave']
            kwargs['fmin'] = self.params['fmin']
        # Plot
        plt.figure(figsize=figsize)
        spec = librosa.amplitude_to_db(np.asarray(np.abs(self))).clip(min=db_thresh)
        librosa.display.specshow(
            data=spec,
            sr=self.sampling_rate,
            cmap=plt.cm.afmhot,
            y_axis='cqt_hz' if self.cqt else 'hz',
            **kwargs
        )
        # Plot formatting
        plt.ylim(fmin, fmax)
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time bins')
        if title is None:
            title = '{}pectrogram from {} to {} Hz'.format(
                'CQT s' if self.cqt else 'S', round(fmin, 1), round(fmax, 1))
        plt.title(title)
        plt.colorbar(format='%+2.0f dB')
        if axis_harm is not None:
            if self.fundamental_freq is not None:
                ytick_freqs = self._harmonic_ticks(fmin, fmax, axis_harm)
                plt.yticks(ytick_freqs)
            else:
                warn('Cannot set harmonics as axis labels because the fundamental '
                     'frequency was not set when creating the spectrogram')
        plt.show()

    def plot_fft(
        self,
        fmin: float = None,
        fmax: float = -1,
        axis_harm: int = None,
        time_bin: int = 0,
        title: str = None,
        figsize: tuple = (15, 4),
        **kwargs
    ):
        """
        Plot a fft as a matplotlib line plot

        Args:
        -----
        fmin: int or float
            Plot minimum frequency
        fmax: int or float
            Plot maximum frequency. None uses 10x the fundamental if it was set,
            otherwise defaults to the nyquist frequency.
        axis_harm: bool or int
            For an STFT: Whether to use harmonic frequencies as y axis labels
            For a CQT: How many harmonic frequencies to use as y axis labels
        time_bin: int
            Which time bin of the spectrogram to plot
        title: str
            Plot title
        figsize: tuple (width: int, height: int)
            Plot dimensions
        **kwargs:
            Matplotlib plot kwargs (e.g ax)
        """
        # Params
        fmax = self._get_fmax(fmax)
        fmin = self._get_fmin(fmin)
        if self.cqt is True:
            freqs = librosa.core.cqt_frequencies(
                self.shape[0], fmin=self.params['fmin'],
                bins_per_octave=self.params['bins_per_octave'])
        else:
            freqs = np.linspace(0, self.nyquist, self.shape[0])
        # Plot
        plt.figure(figsize=figsize)
        plt.plot(
            freqs,
            10 * np.log10(np.abs(self[:, time_bin])),
            **kwargs
        )
        # Plot formatting
        plt.xlim(fmin, fmax)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        if title is None:
            title = '{}FFT at time bin {} from {} to {} Hz'.format(
                    'CQT ' if self.cqt else '', time_bin, round(fmin, 1), round(fmax, 1))
        plt.title(title)
        if self.cqt:
            plt.xscale('log', basex=2)
            axis = plt.gca().xaxis
            axis.set_major_formatter(librosa.display.LogHzFormatter())
            axis.set_major_locator(librosa.display.LogLocator(base=2.0))
            axis.set_minor_formatter(librosa.display.LogHzFormatter(major=False))
        if axis_harm is not None:
            if self.fundamental_freq is not None:
                xtick_freqs = self._harmonic_ticks(fmin, fmax, axis_harm)
                plt.xticks(xtick_freqs)
            else:
                warn('Cannot set harmonics as axis labels because the fundamental '
                     'frequency was not set when creating the spectrogram')
        plt.show()

    def filter_harmonics(
        self,
        neighbour_radius: int = 0
    ):
        """
        Set amplitudes of the spectrogram's non-harmonic frequencies to zero

        Args:
        -----
        neighbour_radius: int
            Number of neighbouring frequencies to keep

        Returns:
        --------
        spectrogram_harmonic: audiolib.Spectrogram
            Modified version of the Spectrogram object
        """
        if self.fundamental_freq is None:
            raise(ValueError(
                'Cannot calculate harmonic frequencies because the fundamental '
                'frequency was not set when creating the spectrogram'))
        if self.cqt is True:
            raise(NotImplementedError('Cannot filter harmonics for CQT spectrograms'))
        spec_harm = np.zeros(self.shape, dtype=self.dtype)
        step = int(self.fundamental_freq * self.shape[0] / self.nyquist)
        for t in range(self.shape[1]):
            for i in range(neighbour_radius+1):
                spec_harm[step+i::step, t] = self[step+i::step, t]
                spec_harm[step-i::step, t] = self[step-i::step, t]
        spec_harm = Spectrogram(
            spec_harm,
            self.sampling_rate,
            self.fundamental_freq,
            self.cqt,
            self.params
        )
        return spec_harm

    def to_audio(self):
        """
        Convert spectrogram to an Audio object

        Returns:
        --------
        recovered_audio: audiolib.Audio
        """
        # Params
        spec = np.asarray(self)
        params = dict(self.params)
        if self.cqt is False:
            if np.issubdtype(self.dtype, np.complexfloating):
                f_inverse = librosa.istft
            else:
                f_inverse = librosa.griffinlim
        else:
            params['sr'] = self.sampling_rate
            del params['n_bins']
            if np.issubdtype(self.dtype, np.complexfloating):
                f_inverse = librosa.icqt
            else:
                f_inverse = librosa.griffinlim_cqt
                filterwarnings("ignore", category=UserWarning)
        # Hacky solution to get librosa inverse functions to work on one time bin
        if self.shape[1] < 2:
            spec = np.hstack([self[:, :1] for _ in range(2)])
            if not self.cqt:
                params['hop_length'] = int(params['hop_length'] * 0.5)
        # Invert audio
        recovered_audio = f_inverse(
            spec,
            **params
        )
        filterwarnings("default", category=UserWarning)
        recovered_audio = audio_.Audio(
            recovered_audio,
            self.sampling_rate,
            self.fundamental_freq
        )
        return recovered_audio

    def _get_fmin(
        self,
        fmin: float
    ):
        """
        Calculate min frequency for plotting
        """
        if fmin is None:
            if self.cqt is True:
                fmin = self.params['fmin']
            else:
                fmin = 0
        return fmin

    def _get_fmax(
        self,
        fmax: float
    ):
        """
        Calculate max frequency for plotting
        """
        if fmax is None:
            if self.fundamental_freq is not None:
                fmax = 10 * self.fundamental_freq
            else:
                fmax = self.nyquist
        elif fmax == -1:
            fmax = self.nyquist
        if self.cqt is True and fmax == self.nyquist:
            num_octaves = (self.params['n_bins'] / self.params['bins_per_octave'])
            fmax = self.params['fmin'] * 2**num_octaves
        return fmax

    def _harmonic_ticks(
        self,
        fmin: float,
        fmax: float,
        axis_harm: int
    ):
        """
        Returns a range of harmonic frequencies to use as plot axis labels

        Args:
        -----
        fmin: int or float
            Minimum frequency of the plot
        fmax: int or float
            maximum frequency of the plot
        axis_harm:
            Number of harmonics to plot (for CQT plot only)
        """
        axis_min = self.fundamental_freq * ((fmin // self.fundamental_freq)+1)
        axis_max = self.fundamental_freq * (fmax // self.fundamental_freq)
        if self.cqt is True:
            axis_max = min(axis_max, self.fundamental_freq * axis_harm)
            axis_step = self.fundamental_freq
        else:
            axis_step = (axis_max - axis_min) / 9
        return range(int(axis_min), int(axis_max)+1, int(axis_step))

    # Convolutions - OUT OF DATE
    def convolve_spectrogram(self):
        """
        Unravel the spectrogram, apply 1-D conv to it then 'ravel' it back
        """
        self._set_harmonic_groups()
        spectrogram_conv = np.ones(self.shape)[::2]
        mask = [1, 1]
        for t in range(self.shape[1]):
            # Time slice to apply convolution to
            spectrogram_slice = self[t]
            # Convolve each group of harmonics
            harmonic_convs = []
            for group in self.harmonic_groups:
                harmonic_amplitudes = spectrogram_slice[group]
                harmonic_convs = np.convolve(mask, harmonic_amplitudes, mode='valid')
                spectrogram_conv[t, group[:-1]] = harmonic_convs

    def _set_harmonic_groups(self):
        """
        Create groups of harmonic frequencies to unwrap the spectrogram
        """
        max_freq = self.shape[0]
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
