import re
import json
from datetime import datetime as dt
from pathlib import Path

from easymake import target
from easymake.helpers import Globals, Shell


g = Globals.from_path('project/src/package/Makefile')
g.DATA_PATH = Path(
    'data/processed/spectrograms/config-${data_config_id}/nsynth-train')
g.OUTPUT_PATH = Path('trainer-output')
g.GCP_CREDENTIALS = Path("credentials/gs-access-key.json")
g.IMAGE_NAME = '${project_name}-${package_name}'
g.BUCKET_NAME = 'deep-musik-data'
g.REGION = 'europe-west1'
g.JOB_ID = 'config${train_config_id}/' + dt.now().strftime('%y%m%d_%H%M%S')

shell = Shell(g, cwd=g.package_path)


# `ezmake local`
@target
def local(id=0):
    """
    Run package locally

    Parameters
    ----------
    id: int
        Training config ID

    """
    g.OUTPUT_PATH /= 'local/${JOB_ID}'
    load_train_config(id)
    shell.run(
        '''
        poetry run python3 -m ${package_name}.task
            --data_dir ${project_path}/${DATA_PATH}
            --job_dir ${project_path}/${OUTPUT_PATH}
            --train_config ${train_config}
            --epochs 10
        '''
    )


# `ezmake docker`
@target
def docker(id=0, r=False):
    """
    Run package locally within a docker container

    Parameters
    ----------
    id: int
        Training config ID
    -r: bool
        Flag to rebuild image

    """
    g.OUTPUT_PATH /= 'docker_local/${JOB_ID}'
    load_train_config(id)
    load_image_uri()
    if r:
        rebuild_image()
    shell.run(
        '''
        docker run --rm
            --volume ${project_path}/data/:/opt/data
            --volume ${project_path}/trainer-output/:/opt/trainer-output
            --name ${IMAGE_NAME}
            ${image_uri}
                --data_dir /opt/${DATA_PATH}
                --job_dir /opt/${OUTPUT_PATH}
                --train_config ${train_config}
                --epochs 10
        '''
    )


# `ezmake docker_gs`
@target
def docker_gs(id=0, r=False):
    """
    Run package locally within a docker container accessing google storage

    Parameters
    ----------
    id: int
        Training config ID
    -r: bool
        Flag to rebuild image

    """
    g.OUTPUT_PATH /= 'docker_local_gs/${JOB_ID}'
    load_train_config(id)
    load_image_uri()
    if r:
        rebuild_image()
    shell.run(
        '''
        docker run --rm
            --volume ${project_path}/credentials/:/opt/credentials/:ro
            --env GOOGLE_APPLICATION_CREDENTIALS=${GCP_CREDENTIALS}
            --name ${IMAGE_NAME}
            ${image_uri}
                --data_dir gs://${BUCKET_NAME}/${DATA_PATH}
                --job_dir gs://${BUCKET_NAME}/${OUTPUT_PATH}
                --train_config ${train_config}
                --epochs 10
        '''
    )


# `ezmake gcloud`
@target
def gcloud(id=-1, r=False, p=False):
    """
    Run package within a docker container on google cloud AI platform

    Parameters
    ----------
    id: int
        Training config ID
    -r: bool
        Flag to rebuild image
    -p: bool
        Flag to push image to GCR

    """
    load_train_config(id)
    # AI platform job variables
    g.OUTPUT_PATH /= '${JOB_ID}'
    g.job_name = g.JOB_ID.replace('/', '_')
    # Load docker image URI: -r rebuilds the image and -p pushes it to GCR
    load_image_uri()
    if r:
        rebuild_image()
    if p:
        shell.run('docker push ${image_uri}')
    # Run on gcloud
    shell.run(
        '''
        gcloud ai-platform jobs submit training ${job_name}
            --region ${REGION}
            --master-image-uri ${image_uri}
            --config ${package_path}/config/hparams.yaml
            --
                --data_dir gs://${BUCKET_NAME}/${DATA_PATH}
                --job_dir gs://${BUCKET_NAME}/${OUTPUT_PATH}
                --train_config ${train_config}
                --epochs 1
        '''
    )


# `ezmake clean`
@target
def clean():
    """Remove trainer local output (tensorboard and model checkpoints)"""
    print('Removing trainer output at:', g.project_path / g.OUTPUT_PATH)
    shell.run('sudo rm -rf ${project_path}/${OUTPUT_PATH}')


# `ezmake tests`
@target
def tests(*args):
    """Run trainer tests"""
    shell.run('.venv/bin/python3 -m pytest')


# Helper functions

def load_train_config(config_id=-1):
    """
    Load JSON config for the training job

    Parameters
    ----------
    config_id: int
        Training config ID

    """
    # Parse JSON
    with open('config/train_configs.json', 'r') as f:
        configs = json.load(f)
    last_id = configs[-1]['id']
    # Request user input: numeric config between 0 and last_id
    while not (0 <= config_id <= last_id):
        config_id = input(f'Enter training config ID (0 to {last_id}): ')
        config_id = int(config_id) if re.findall(r'^\d+$', config_id) else -1
    # Parse training config
    g.train_config_id = config_id
    g.train_config = configs[config_id]['config']
    g.data_config_id = g.train_config['data_config_id']


def load_image_uri():
    """Convert image name to a URI for gcloud container registry"""
    g.gcp_project_name = shell.capture(
        'gcloud config list project --format "value(core.project)"')
    g.image_tag = 'latest'
    g.image_uri = 'eu.gcr.io/${gcp_project_name}/${IMAGE_NAME}:${image_tag}'


def rebuild_image():
    shell.run('zsh ${package_path}/docker/docker-build.zsh')
