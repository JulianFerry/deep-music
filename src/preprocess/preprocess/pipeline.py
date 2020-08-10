import os

import apache_beam as beam
from apache_beam.io.fileio import MatchAll, ReadMatches, WriteToFiles
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from .preprocess import load_audio, PreProcessor
from .pickle import PickleSink, pickle_naming


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
