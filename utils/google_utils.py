# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# This file contains google utils: https://cloud.google.com/storage/docs/reference/libraries
# pip install --upgrade google-cloud-storage

import os
import time

# from google.cloud import storage


def gdrive_download(id="1HaXkef9z6y5l4vUnCYgdmEAj61c6bfWO", name="coco.zip"):
    """Downloads a file from Google Drive using its ID and saves it with the provided name, unzipping if necessary."""
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()
    t = time.time()

    print(f"Downloading https://drive.google.com/uc?export=download&id={id} as {name}... ", end="")
    if os.path.exists(name):  # remove existing
        os.remove(name)

    # Attempt large file download
    s = [
        f'curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={id}" > /dev/null',
        f"curl -Lb ./cookie -s \"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {{print $NF}}' ./cookie`&id={id}\" -o {name}",
        "rm ./cookie",
    ]
    r = sum(os.system(x) for x in s)

    # Attempt small file download
    if not os.path.exists(name):  # file size < 40MB
        s = f"curl -f -L -o {name} https://drive.google.com/uc?export=download&id={id}"
        r = os.system(s)

    # Error check
    if r != 0:
        os.system(f"rm {name}")
        print("ERROR: Download failure ")
        return r

    # Unzip if archive
    if name.endswith(".zip"):
        print("unzipping... ", end="")
        os.system(f"unzip -q {name}")
        os.remove(name)  # remove zip to free space

    print("Done (%.1fs)" % (time.time() - t))
    return r


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to a bucket: https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python."""
    # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from a Google Cloud Storage bucket to a local file."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
