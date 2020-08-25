import re
import pickle

from apache_beam.io.fileio import FileSink


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
    file_name = re.findall('(^.*).wav', wav_name)[0] + '.spec'
    return file_name


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
