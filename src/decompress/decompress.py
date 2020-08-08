import io
import os
import tarfile
from flask import Flask, request
from google.cloud import storage

import logging
from logging.config import dictConfig
import traceback

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials/gs-access-key.json'
dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] [%(levelname)s] %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})
app = Flask(__name__)
app.logger.setLevel(logging.INFO)


def warning(msg):
    """
    Create app warning including exception traceback
    """
    msg = 'Error: ' + msg
    app.logger.warning(msg + '\n' + traceback.format_exc())
    return msg

@app.route('/', methods=['GET', 'POST'])
def untar():
    """
    Untars files as specified by bucket_name, zip_path, save_path

    Parameters
    ----------
    --data: POST request json data
        Should include "bucket_name", "zip_path", "save_path"

    Example
    -------
    curl -X POST 0.0.0.0:8008 \
        --header "Content-Type: application/json" \
        --data '{ \
            "bucket_name": "mybucket", \
            "zip_path": "bucket/path/to/tar.gz", \
            "save_path": "bucket/output/dir" \
        }'

    """
    if request.method == 'GET':
        return 'Application is running'

    elif request.method == 'POST':
        try:
            bucket_name = request.json['bucket_name']
            zip_path = request.json['zip_path']
            save_path = request.json['save_path']
        except:
            return warning('POST request must contain bucket_name, zip_path and save_path')

        # Connect to cloud storage
        client = storage.Client()
        try:
            bucket = client.get_bucket(bucket_name)
        except:
            return warning(f'bucket_name incorrect ("{bucket_name}")')

        # Download tar.gz file
        app.logger.info(f'Downloading {zip_path} to disk')
        tmp_path = 'data/tmp.tar.gz'
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        with open(tmp_path, 'wb') as file_obj:
            try:
                bucket.get_blob(zip_path).download_to_file(file_obj)
            except:
                return warning(f'zip_path incorrect: "{zip_path}"')

        # Decompress and upload to cloud
        with open('data/tmp.tar.gz', 'rb') as file_obj:
            tar = tarfile.open(fileobj=file_obj, mode='r:gz')
            # Iterate over all files in the tar folder
            file_count = len(tar.getnames())
            progress_tick = int(file_count * 0.05)
            app.logger.info(f'Decompressing {file_count} files from {zip_path}')
            for i, file_name in enumerate(tar.getnames()):
                if i % progress_tick == 0:
                    progress = int(100 * (i / file_count))
                    app.logger.info(f'{progress}% complete ({i}/{file_count})')
                file_object = tar.extractfile(file_name)
                if file_object:
                    output_blob = bucket.blob(os.path.join(save_path, file_name))
                    output_blob.upload_from_string(file_object.read())
        msg = f'Success: Saved decompressed files to "{save_path}"'
        app.logger.info(msg)
        return msg

    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8008)