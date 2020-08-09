import io
import os
import tarfile
from google.cloud import storage


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials/gs-access-key.json'


def untar(bucket_name, zip_dir, zip_file, save_dir):
    """
    Untars files as specified by bucket_name, zip_path, save_dir

    Parameters
    ----------
    bucket_name: str
        Cloud storage bucket used to load and save data
    zip_dir: str
        Bucket path to the tar.gz file\'s parent directory
    zip_file: str
        Name of the .tar.gz zip file
    save_dir: str
        Bucket path to the directory in which to save decompressed files

    """
    # Connect to cloud storage
    client = storage.Client()
    try:
        bucket = client.get_bucket(bucket_name)
    except:
        raise ValueError(f'bucket_name incorrect ("{bucket_name}")')

    # Download tar.gz file
    zip_path = os.path.join(zip_dir, zip_file)
    print(f'Downloading {zip_path} to disk')
    tmp_path = 'data/tmp.tar.gz'
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    with open(tmp_path, 'wb') as file_obj:
        try:
            bucket.get_blob(zip_path).download_to_file(file_obj)
        except:
            raise ValueError(f'zip_path incorrect: "{zip_path}"')

    # Decompress and upload to cloud
    with open('data/tmp.tar.gz', 'rb') as file_obj:
        tar = tarfile.open(fileobj=file_obj, mode='r:gz')
        file_count = len(tar.getnames())
        print(f'Decompressing {file_count} files from {zip_path}')
        # Iterate over all files in the tar folder
        for i, file_name in enumerate(tar.getnames()):
            percent = int(100 * (i / file_count))
            print(f'{percent}% complete ({i}/{file_count})', end='\r')
            file_object = tar.extractfile(file_name)
            if file_object:
                output_blob = bucket.blob(os.path.join(save_dir, file_name))
                output_blob.upload_from_string(file_object.read())
    print(f'\nSuccess: Saved decompressed files to "{save_dir}"')