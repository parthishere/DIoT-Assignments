from google.cloud import storage
import os

"""
1) Create GCP account and project
2) Enable Cloud Storage API
3) Create a non-public storage bucket
4) Create service account with Storage Admin permissions
5) Download JSON credentials file
"""


# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./utils/service_account.json"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_name)
    
    return f"gs://{bucket_name}/{destination_blob_name}"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    blob.download_to_filename(destination_file_name)
    
    return destination_file_name