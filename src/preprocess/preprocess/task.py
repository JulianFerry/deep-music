import os
import argparse
import json
import logging
from warnings import warn

import apache_beam as beam
from apache_beam.io.fileio import MatchAll, ReadMatches, WriteToFiles
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from .preprocess import load_audio, PreProcessor
from .pickle import PickleSink, pickle_naming


def get_args():
    """
    Argument parser.
    See preprocessing.preprocess_dataset modoule for more info.

    Returns
    -------
    Dictionary of arguments

    """
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument(
        '--data_dir',
        help='Path to the dataset of audio .wav files',
        type=str
    )
    parser.add_argument(
        '--filters_dir',
        help='Path to the instrument filter files',
        type=str
    )
    parser.add_argument(
        '--job_dir',
        help='Where to save the spectrograms (root directory)',
        type=str
    )

    # Preprocessing arguments
    parser.add_argument(
        '--config',
        help='Preprocessing options',
        type=json.loads,
        default='{}'
    )
    parser.add_argument(
        '--instruments',
        help='Instruments to apply preprocessing to',
        type=json.loads,
        default='[]'
    )

    # Parse
    known_args, pipeline_args = parser.parse_known_args()
    # pipeline_args.extend([
    #     '--runner=DataFlowRunner',
    #     '--project="deep-musik"',
    #     '--region=SET_REGION_HERE',
    #     '--staging_location=gs://YOUR_BUCKET_NAME/AND_STAGING_DIRECTORY',
    #     '--temp_location=gs://YOUR_BUCKET_NAME/AND_TEMP_DIRECTORY',
    #     '--job_name=your-wordcount-job',
    # ])
    return known_args.__dict__, pipeline_args


def run_pipeline(args, pipeline_args, save_main_session=True):
    """

    """

    # Handle args: config
    pp = PreProcessor(fft_params=args['config'])
    #pp.save_config(args['job_dir'])

    # Handle args: instruments, filters_dir, exclude
    instrument_ids = {'guitar_acoustic': ['guitar_acoustic_029-069']}
    #instrument_ids = filter_instrument_ids(args['filters_dir'])
    #exclude = args['exclude'] # make this a separate config argument
    instrument_ids = [id for instr in args['instruments'] for id in instrument_ids.get(instr, [])]
    def id_to_path(id):
        return os.path.join(args['data_dir'], 'audio', id + '*.wav')

    # Handle args: job_dir
    def return_destination(*args):
        return args[-1]
    pickle_writer = WriteToFiles(
        path=args['job_dir'],
        file_naming=return_destination,
        destination=pickle_naming,
        sink=PickleSink()
    )
    
    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g. a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    # Beam Pipeline
    with beam.Pipeline(options=pipeline_options) as p:
        (p
            | 'LoadInstrumentIds' >> beam.Create(instrument_ids)
            | 'CreateFilePaths' >> beam.Map(id_to_path)
            | 'MatchFiles' >> MatchAll()
            | 'OpenFiles' >> ReadMatches()
            | 'LoadAudio' >> beam.Map(load_audio)
            | 'AudioToSpectrogram' >> beam.Map(pp.audio_to_spectrogram)
            | 'SaveSpectrogram' >> pickle_writer
        )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = get_args()
    run_pipeline(*args)