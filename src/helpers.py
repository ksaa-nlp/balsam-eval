# A function to normalize strings to id like format (normalized and sanitized to be used as a file name too)
import unicodedata
"""A module for working with tasks and datasets."""

import json
import os
from typing import Any


from google.cloud import storage

def normalize_string(text: str) -> str:
    return (
        unicodedata.normalize("NFKC", text)
        .lower()
        .replace("\x00", "")
        .strip()
        [:255]
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
    )
    
def download_dataset_from_gcs(dataset_id: str, directory: str) -> dict[str, Any]:
    """
    Download a dataset from GCS. The dataset is identified by a dataset ID
    which corresponds to its primary key in the database.
    """
    storage_client = storage.Client()
    bucket_name = os.getenv("GCLOUD_BUCKET")
    bucket = storage_client.bucket(bucket_name)
    # Create a temporary directory to store the dataset
    os.makedirs(directory, exist_ok=True)
    blob = bucket.blob(f"datasets/{dataset_id}.json")
    blob.download_to_filename(f".temp/{dataset_id}.json")

    print(f"Downloaded {dataset_id}.json from GCS bucket.")

    # Read the dataset from the file
    with open(f".temp/{dataset_id}.json", "r", encoding="utf8") as fp:
        dataset = json.load(fp)
        dd = dataset["json"]
        dd["task"] = dataset["task"]
        dd["category"] = dataset["category"]
        # Overwrite the dataset with the new data
        with open(f".temp/{dataset_id}.json", "w", encoding="utf8") as fp:
            json.dump(dd, fp, ensure_ascii=False)
        return dd


