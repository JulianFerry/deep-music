import os
import re
import json
from pathlib import Path

from easymake import target
from easymake.helpers import Globals, Shell


g = Globals.from_path('project/src/package/Makefile')
g.DATA_PATH = Path('data/raw/nsynth-${dataset}')
g.FILTERS_PATH = Path('data/interim/filters/nsynth-${dataset}')
g.OUTPUT_DIR = Path('data/processed/spectrograms')
g.OUTPUT_PATH = Path(
    '${OUTPUT_DIR}/config-${data_config_id}/nsynth-${dataset}')
g.BUCKET_NAME = 'deep-musik-data'
g.REGION = 'europe-west1'
g.GCP_CREDENTIALS = '${project_path}/credentials/gs-access-key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = g.GCP_CREDENTIALS

shell = Shell(g, cwd=g.package_path)


# `ezmake local`
@target
def local(
    id=0,
    datasets='train',
    instruments=["brass_electronic", "string_electronic"]
):
    """
    Run package locally

    Parameters
    ----------
    id: int
        Data preprocessing config ID
    datasets: str
        Datasets to train on (train/valid/test)
    instruments: str
        JSON list of instruments to preprocess data for

    """
    build_audiolib()
    load_data_config(config_id=id)
    input_datasets(datasets=datasets)
    get_instruments(instruments=instruments)
    for g.dataset in g.datasets:
        shell.run(
            '''
            .venv/bin/python preprocessing_main.py
                --data_dir ${project_path}/${DATA_PATH}
                --filters_dir ${project_path}/${FILTERS_PATH}
                --job_dir ${project_path}/${OUTPUT_PATH}
                --config ${data_config}
                --instruments ${instruments}
            '''
        )


# `ezmake dataflow`
@target
def dataflow(
    id=-1,
    datasets='',
    instruments=["brass_electronic", "string_electronic"]
):
    """
    Run package on dataflow

    Parameters
    ----------
    id: int
        Data preprocessing config ID
    datasets: str
        Datasets to train on (train/valid/test)
    instruments: str
        JSON list of instruments to preprocess data for

    """
    build_audiolib()
    load_data_config(config_id=id)
    input_datasets(datasets=datasets)
    get_instruments(instruments=instruments)
    shell.run('''gcloud auth activate-service-account
                --key-file ${GCP_CREDENTIALS}''')
    for g.dataset in g.datasets:
        shell.run(
            '''
            .venv/bin/python preprocessing_main.py
                --data_dir gs://${BUCKET_NAME}/${DATA_PATH}
                --filters_dir gs://${BUCKET_NAME}/${FILTERS_PATH}
                --job_dir gs://${BUCKET_NAME}/${OUTPUT_PATH}
                --config ${data_config}
                --instruments ${instruments}
                --runner dataflow
                --num_workers 1
                --noauth_local_webserver
                --setup_file ${package_path}/setup.py
                --extra_package ${package_path}/audiolib-0.1.0.tar.gz
            '''
        )


# `ezmake upload_filters`
@target
def upload_filters():
    """Upload manually created data filters to google storage"""
    shell.run('''
        gsutil -m cp -r
        ${project_path}/data/interim/filters gs://${BUCKET_NAME}/data/interim/
    ''')


# `ezmake clean`
@target
def clean():
    """Delete processed spectrograms in the local output directory"""
    print('Removing preprocessed data at:', g.project_path / g.OUTPUT_DIR)
    shell.run('rm -r ${project_path}/${OUTPUT_DIR}')


# `ezmake tests`
@target
def tests(*args):
    """Run preprocessing tests"""
    shell.run('.venv/bin/python3 -m pytest')


# Helper functions

def load_data_config(config_id=-1):
    """
    Load JSON config for the preprocessing job

    Parameters
    ----------
    i: int
        Data preprocessing config ID
    d: str
        Dataset to train on (train/valid/test)

    """
    # Parse JSON
    with open('config/data_configs.json', 'r') as f:
        configs = json.load(f)
    last_id = configs[-1]['id']
    # Request user input: numeric config between 0 and last_id
    while not (0 <= config_id <= last_id):
        config_id = input(f'Enter preprocessing config ID (0 to {last_id}): ')
        config_id = int(config_id) if re.findall(r'^\d+$', config_id) else -1
    # Parse preprocesing config
    g.data_config_id = config_id
    g.data_config = configs[config_id]['config']


def input_datasets(datasets):
    """
    Request user input: datasets to preprocess (train/valid/test)
    """
    if isinstance(datasets, list):
        datasets = ' '.join(datasets)
    sets = ['train', 'valid', 'test']
    while not all([s.strip(' ') in sets for s in datasets.split(' ')]):
        datasets = input('Enter dataset(s) to preprocess '
                         '(train/valid/test) separated by spaces: ')
    g.datasets = datasets.split(' ')


def get_instruments(instruments):
    """
    Parse instrument input
    """
    if isinstance(instruments, str):
        instruments = instruments.split(' ')
    g.instruments = json.dumps(instruments)


def build_audiolib():
    """
    Download and build the source distribution of audiolib (not on PyPi)
    for the Dataflow ``--extra_package`` argument
    """
    tar_file = 'audiolib-0.1.0.tar.gz'
    if not os.path.exists(g.package_path/tar_file):
        print('Building audiolib-0.1.0 as a local dependency for Dataflow')
        git_repo = 'ssh://git@github.com/JulianFerry/audiolib.git'
        shell.run(f'git clone -b "0.1.0" --single-branch --depth 1 {git_repo}')
        shell.run(
            'poetry build -f sdist',
            cwd=g.package_path / 'audiolib'
        )
        shell.run(f'''mv audiolib/dist/{tar_file} .;  rm -rf audiolib''')
