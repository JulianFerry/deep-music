import os
import subprocess

AUTHENTICATED = False


def gcloud_auth():
    """
    Authenticate with local GCP credentials if they have been mounted
    """
    key_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    if os.path.exists(key_file):
        subprocess.run(['gcloud', 'auth', 'activate-service-account',
                        '--key-file', key_file])
    global AUTHENTICATED
    AUTHENTICATED = True


def upload(local_path, gs_path):
    if not AUTHENTICATED:
        gcloud_auth()
    subprocess.run(['gsutil', '-m', 'cp', '-r', local_path, gs_path])


def download(gs_path, local_path):
    if not AUTHENTICATED:
        gcloud_auth()
    subprocess.run(['gsutil', '-m', 'cp', '-r', gs_path+'**', local_path])
