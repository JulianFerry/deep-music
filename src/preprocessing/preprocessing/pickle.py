import os
import re
import pickle

from apache_beam.io.fileio import FileSink


class PickleSink(FileSink):
    """
    A sink to a GCS or local text file or files.

    This sink simply calls file_handler.write(pickle.dumps(record)) on
    all records that come into it.

    """
    def open(self, fh):
        self._fh = fh

    def write(self, record):
        file_name, data = record
        self._fh.write(pickle.dumps(data))

    def flush(self):
        self._fh.flush()


def pickle_naming(record: tuple):
    """
    Generate a spectrogram pickle name from the original wav file name.

    Parameters
    ----------
    record: tuple
        Beam PCollection record

    Returns
    -------
    str:
        Spectrogram pickle file name

    """
    wav_name, spectrogram = record
    instr_id = wav_name.split('-')[0]
    instr_name = re.findall('(^.*)_', wav_name)[0]
    file_name = re.findall('(^.*).wav', wav_name)[0] + '.spec'
    file_path = os.path.join(instr_name, instr_id, file_name)
    return file_name