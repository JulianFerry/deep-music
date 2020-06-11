import argparse
import os
import subprocess
from . import model


def get_args():
    """
    Argument parser

    Returns
    -------
    Dictionary of arguments

    """
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument(
        '--epochs',
        help='Epochs to run the training job for',
        type=int,
        default=1
    )

    # Paths
    parser.add_argument(
        '--job_dir',
        help='GCS location to write logs and checkpoint model weights',
    )
    parser.add_argument(
        '--data_dir',
        help='GCS location to fetch data from',
    )

    # Parse
    args = parser.parse_args()
    args = args.__dict__

    # Mount data from cloud storage bucket if necessary
    if args.get('data_dir', '').startswith('gs://'):
            bucket_name = args['data_dir'][5:].split('/')[0]
            cmd_gcsfuse = ['gcsfuse', '--implicit-dirs', bucket_name, '/root/data']
            # Use local GCP credentials if they have been mounted
            if os.path.isdir('/root/credentials'):
                cmd_gcsfuse.insert(-2, '--key-file')
                cmd_gcsfuse.insert(-2, '/root/credentials/gs-access-key.json')
            subprocess.run(cmd_gcsfuse)
            split_path = args['data_dir'][5:].split('/')[1:]
            args['data_dir'] = '/'.join(['/root/data'] + split_path)
 
    return args


if __name__ == '__main__':
    # Parse command-line arguments
    args = get_args()
    # Run the training job
    model.train_and_evaluate(args)
