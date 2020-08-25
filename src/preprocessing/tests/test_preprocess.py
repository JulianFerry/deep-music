from preprocessing.preprocess import load_audio, PreProcessor

import os
import json
from audiolib import Audio
from audiolib import samples

import apache_beam as beam
from apache_beam.io.fileio import MatchAll, ReadMatches
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to


DIR = os.path.abspath(os.path.dirname(__file__))
test_file = os.path.join(DIR, 'files/bass_electronic.wav')


def test_load_audio():
    """Test that pipeline loads files as audiolib.Audio objects"""
    with TestPipeline() as p:
        files = (
            p
            | beam.Create([test_file])
            | MatchAll()
            | ReadMatches()
        )
        output = (
            files
            | beam.Map(load_audio)
            | beam.Map(lambda x: (x[0], type(x[1]), x[1].shape))
        )

        assert_that(
            output,
            equal_to([('bass_electronic.wav', Audio, (64000, ))])
        )


class TestPreProcessor:
    params = {
        'time_intervals': 5,
        'resolution': 5,
        'start': 0,
        'end': -1,
        'exclude': []
    }
    pp = PreProcessor(params)
    record = ('piano', Audio(*samples.piano()))

    def test_init(self):
        """Test that preprocessor loads the config"""
        assert self.pp.fft_params == self.params

    def test_audio_to_spectrogram(self):
        """Test that preprocessor converts audio to spectrograms"""
        record = self.pp.audio_to_spectrogram(self.record)

        assert type(record) == tuple
        assert record[0] == self.record[0]
        assert record[1].shape == (476, 5)
        assert record[1].cqt is True

    def test_save_config(self, tmpdir):
        """Test that preprocessor saves the configuration as JSON"""
        self.pp.save_config(tmpdir.strpath)
        config_path = os.path.join(tmpdir, 'config.json')

        assert os.path.exists(config_path)
        with open(config_path) as f:
            config = json.load(f)
        assert isinstance(config, dict)
        assert config == self.params
