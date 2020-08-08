import io
import os
import tarfile
from google.cloud import storage


def untar(request):
    """
    
    """
    request_json = request.get_json()
    bucket_name = request_json['bucket_name']
    zip_path = request_json['zip_path']
    save_path = request_json['save_path']
    
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    # Open tar.gz file
    with open('/tmp/tar.gz', 'wb') as file_obj:
        bucket.get_blob(zip_path).download_to_file(file_obj)
    with open('/tmp/tar.gz', 'rb') as file_obj:
        tar = tarfile.open(fileobj=file_obj, mode='r:gz')
        # Iterate over all files in the tar file
        for file_name in tar.getnames():
            file_object = tar.extractfile(file_name)
            if file_object:
                output_blob = bucket.blob(os.path.join(save_path, file_name))
                output_blob.upload_from_string(file_object.read())


# gcloud beta functions deploy untar \
#     --runtime python37 \
#     --project deep-musik \
#     --trigger-http \
#     --region europe-west1 \
#     --timeout 540 \
#     --memory 512MB

# gcloud functions call untar \
#     --region europe-west1 \
#     --data '{"bucket_name": "deep-musik-data", "zip_path": "download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz", "save_path": "data/raw/"}'