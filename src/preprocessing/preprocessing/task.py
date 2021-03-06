import sys
import argparse
import json

from .pipeline import run_pipeline


def parse_args(args):
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
    known_args, pipeline_args = parser.parse_known_args(args)
    known_args = known_args.__dict__
    if '--runner' in pipeline_args:
        runner_idx = pipeline_args.index('--runner')
        if len(pipeline_args) > runner_idx + 1:
            if pipeline_args[runner_idx+1] == 'dataflow':
                pipeline_args.extend([
                    '--project=deep-musik',
                    '--region=europe-west1',
                    '--staging_location=gs://deep-musik-data/staging',
                    '--temp_location=gs://deep-musik-data/tmp',
                    '--job_name=preprocess',
                ])
    return known_args, pipeline_args


def run():
    args = parse_args(sys.argv[1:])
    run_pipeline(*args)
