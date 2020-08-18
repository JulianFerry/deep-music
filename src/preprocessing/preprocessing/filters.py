import os
import pandas as pd


def filter_instrument_ids(filters_dir, instruments):
    """
    Fetch instrument filters from the path and return the IDs to preprocess

    Parameters
    ----------
    filters_path: str
        Root directory containing instrument filter CSVs
    instruments: list
        List of instruments to return IDs for

    """
    if filters_dir is None:
        return instruments
    instr_ids = []
    for instr in instruments:
        filter_csv = os.path.join(filters_dir, instr + '-filter.csv')
        df_filter = pd.read_csv(filter_csv)
        ids_to_keep = df_filter.loc[df_filter['keep'] == 1, 'instrument'].values
        instr_ids.extend(list(ids_to_keep))
    return instr_ids


class FileFilter:

    def __init__(self, data_dir, config):
        """
        Load metadata for all files
        """
        metadata = pd.read_json(
            os.path.join(data_dir, 'examples.json'),
            orient='index'
        )
        self.qualities = metadata['qualities_str']
        self.exclude = set(config.get('exclude', []))

    def filter_audio_qualities(self, record):
        """
        Parameters
        ----------
        record:
            File names returned by beam.MatchAll()

        """
        instr_id = record.path.split('/')[-1].replace('.wav', '')
        return len(self.exclude.intersection(self.qualities[instr_id])) == 0