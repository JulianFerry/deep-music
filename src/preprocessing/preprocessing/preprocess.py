import audiolib

def audio_to_spectrogram(
    audio: audiolib.audio.Audio,
    fft_params: dict
):
    """
    Convert an audio object to a spectrogram given a set of fft_params
    """
    spec = (audio
        .trim(fft_params['start'], fft_params['end'])
        .to_spectrogram(
            fft_params['time_intervals'],
            fft_params['resolution'],
            cqt=True
        ))
    return spec

