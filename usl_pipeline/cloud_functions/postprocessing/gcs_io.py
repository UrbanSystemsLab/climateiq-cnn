"""GCS download and upload helpers."""

import logging
import os

from google.cloud import storage


def _download_from_gcs(bucket_name, source_blob, dest_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(dest_path)
    logging.info("Downloaded gs://%s/%s", bucket_name, source_blob)


def _upload_to_gcs(bucket_name, source_path, dest_blob):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(source_path)
    logging.info("Uploaded to gs://%s/%s", bucket_name, dest_blob)


def _download_directory_from_gcs(bucket_name, prefix, dest_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    os.makedirs(dest_dir, exist_ok=True)
    for blob in blobs:
        if not blob.name.endswith("/"):
            relative_path = blob.name[len(prefix) :].lstrip("/")
            dest_path = os.path.join(dest_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            blob.download_to_filename(dest_path)
            logging.info("Downloaded %s", blob.name)
