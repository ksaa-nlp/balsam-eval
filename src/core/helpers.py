"""Helper utilities for dataset normalisation and GCS object IO."""

import logging
import os
import unicodedata

from google.cloud import storage

logger = logging.getLogger(__name__)


def normalize_string(text: str) -> str:
    """Normalise a string for use as an ID or filename."""
    return (
        unicodedata.normalize("NFKC", text)
        .lower()
        .replace("\x00", "")
        .strip()[:255]
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def sanitize_config_name(name: str) -> str:
    """Strip characters forbidden by the HuggingFace ``datasets`` library."""
    forbidden = "<>:/\\|?*"
    sanitized = name
    for char in forbidden:
        sanitized = sanitized.replace(char, "_")
    return sanitized


def download_pool_file_from_gcs(
    bucket: str,
    object_path: str,
    dest_path: str,
) -> None:
    """Download a single GCS object to ``dest_path``.

    Args:
        bucket: GCS bucket name.
        object_path: Object path within the bucket.
        dest_path: Local file path to write to (parent dirs are created).
    """
    client = storage.Client()
    blob = client.bucket(bucket).blob(object_path)
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    blob.download_to_filename(dest_path)
    logger.info("Downloaded gs://%s/%s -> %s", bucket, object_path, dest_path)


def upload_result_file_to_gcs(
    bucket: str,
    local_path: str,
    object_path: str,
) -> None:
    """Upload a local JSON result file to ``gs://bucket/object_path``."""
    client = storage.Client()
    blob = client.bucket(bucket).blob(object_path)
    blob.upload_from_filename(
        local_path, content_type="application/json; charset=utf-8"
    )
    logger.info("Uploaded %s -> gs://%s/%s", local_path, bucket, object_path)
