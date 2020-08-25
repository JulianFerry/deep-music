import time
from easymake import target
from easymake.helpers import Globals, Shell


g = Globals.from_path('project/src/package/Makefile')
g.BUCKET_NAME = 'deep-musik-data'
g.IMAGE_NAME = '${project_name}-${package_name}'

shell = Shell(g, cwd=g.package_path)


# `ezmake local`
@target
def local(*datasets):
    """Run the package locally"""
    input_datasets(datasets)
    shell.run(
        'sh docker/entrypoint.sh ${project_path} ${BUCKET_NAME} ${datasets}',
        cwd=g.package_path
    )


# `ezmake docker`
@target
def docker(r=False, *datasets):
    """Run the docker container locally"""
    input_datasets(datasets)
    load_image_uri()
    if r:
        rebuild_image()
    shell.run(
        '''
        docker run --rm --name ${IMAGE_NAME} $image_uri
            /home ${BUCKET_NAME} ${datasets}
        ''',
        cwd=g.package_path
    )


# `ezmake gcloud`
@target
def gcloud(r=False, p=False, *datasets):
    """Run the docker container on a google cloud VM"""
    input_datasets(datasets)
    load_image_uri()
    if r:
        rebuild_image()
    if p:
        shell.run('docker push ${image_uri}')
    g.metadata = ','.join([
        "image_name=${image_uri}",
        "container_args=/home ${BUCKET_NAME} ${datasets}"
    ])
    shell.run(
        '''
        gcloud compute instances create ${package_name}
            --image-family=cos-stable
            --image-project=cos-cloud
            --boot-disk-size 60G
            --scopes compute-rw,logging-write,storage-rw
            --metadata ${metadata}
            --metadata-from-file startup-script=${package_path}/startup-script.sh
        '''
    )
    print('\nSSH to compute instance:', g.package_name)
    time.sleep(5)
    shell.run('gcloud compute ssh ${package_name}')


# Helper functions

def input_datasets(datasets):
    """User input: datasets to download (train/valid/test)"""
    if isinstance(list, datasets):
        datasets = ' '.join(datasets)
    while not any([s in datasets for s in ['train', 'valid', 'test']]):
        datasets = input(
            'Enter dataset(s) to download (train/valid/test) separated by spaces: ')
    g.datasets = datasets


def load_image_uri():
    """Convert image name to a URI for gcloud container registry"""
    g.gcp_project_name = shell.capture(
        'gcloud config list project --format "value(core.project)"')
    g.image_tag = 'latest'
    g.image_uri = 'eu.gcr.io/${gcp_project_name}/${IMAGE_NAME}:${image_tag}'


def rebuild_image():
    shell.run('zsh ${package_path}/docker/docker-build.zsh')
