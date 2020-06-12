import os
import subprocess

AUTHENTICATED = False

def gcloud_auth():
    """
    Authenticate with local GCP credentials if they have been mounted
    """
    if os.path.isdir('/root/credentials'):
        key_path = '/root/credentials/gs-access-key.json'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
        subprocess.run(['gcloud', 'auth', 'activate-service-account', '--key-file', key_path])
    global AUTHENTICATED
    AUTHENTICATED=True


def upload(local_path, gs_path):
    if not AUTHENTICATED:
        gcloud_auth()
    subprocess.run(['gsutil', '-m', 'cp', '-r', local_path, gs_path])


def download(gs_path, local_path):
    if not AUTHENTICATED:
        gcloud_auth()
    subprocess.run(['gsutil', '-m', 'cp', '-r', gs_path, local_path])