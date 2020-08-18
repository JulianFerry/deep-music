import os

import apache_beam as beam
from apache_beam.io.fileio import MatchAll, ReadMatches, WriteToFiles
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from .preprocess import load_audio, PreProcessor
from .filters import filter_instrument_ids, FileFilter
from .pickle import PickleSink, pickle_naming


def run_pipeline(args, pipeline_args, save_main_session=True):
    """

    """
    # File input args: data_dir
    def id_to_path(id):
        return os.path.join(args['data_dir'], 'audio', id + '*.wav')

    # File filtering args: instruments, filters_dir, config['exclude']
    instrument_ids = filter_instrument_ids(args.get('filters_dir'), args['instruments'])
    ff = FileFilter(args['data_dir'], args['config'])

    # Preprocessing args: config
    pp = PreProcessor(fft_params=args['config'])
    pp.save_config(args['job_dir'])

    # Output args: job_dir
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
            | 'FilterFiles' >> beam.Filter(ff.filter_audio_qualities)
            | 'OpenFiles' >> ReadMatches()
            | 'LoadAudio' >> beam.Map(load_audio)
            | 'AudioToSpectrogram' >> beam.Map(pp.audio_to_spectrogram)
            | 'SaveSpectrogram' >> pickle_writer
        )
