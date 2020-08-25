from preprocessing.pickle import pickle_naming, PickleSink

import os
import json
import numpy as np
import pickle

import apache_beam as beam
from apache_beam.io.fileio import WriteToFiles
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from numpy.testing import assert_array_equal


DIR = os.path.abspath(os.path.dirname(__file__))
test_file = os.path.join(DIR, 'files/bass_electronic.wav')


def test_pickle_naming():
    file_name = 'my_instr_001-002-003.wav'
    spec_name = pickle_naming((file_name + '.wav', 0))
    assert spec_name == file_name + '.spec'


class TestPickleSink:

    def test_write(self, tmpdir):
        """Test that pipeline writes to pickle"""
        file_name = 'numpy_random'
        array = np.random.random(100)
        pickle_writer = WriteToFiles(
            path=tmpdir,
            file_naming=lambda *args: file_name,
            sink=PickleSink()
        )
        with TestPipeline() as p:
            (p
                | beam.Create([(file_name, array)]) 
                | pickle_writer
            )
        with open(os.path.join(tmpdir, file_name), 'rb') as f:
            assert_array_equal(pickle.load(f), array)