"""Common utility functions shared across the project."""

import json
import os
import shutil

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()


def setup_directories(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def copy_multimodal_utils_to_temp(temp_dir: str = ".temp") -> str | None:
    """Copy multimodal_utils.py to temp directory for lm_eval to find it.

    Args:
        temp_dir: Directory to copy the file to

    Returns:
        Path to the copied or generated file.
    """
    multimodal_utils_dst = os.path.join(temp_dir, "multimodal_utils.py")
    source_candidates = [
        "src/core/multimodal_utils.py",
        os.path.join(os.path.dirname(__file__), "multimodal_utils.py"),
    ]

    os.makedirs(temp_dir, exist_ok=True)
    for multimodal_utils_src in source_candidates:
        if not os.path.exists(multimodal_utils_src):
            continue
        shutil.copy2(multimodal_utils_src, multimodal_utils_dst)
        print(f"Copied multimodal_utils.py to {multimodal_utils_dst}")
        return multimodal_utils_dst

    fallback_content = '''"""Standalone multimodal helpers for lm_eval YAML imports."""

import os

from PIL import Image


def doc_to_image(doc):
    images = []
    for path in doc.get("images", []):
        try:
            images.append(Image.open(path))
        except (OSError, IOError) as exc:
            print(f"Warning: Failed to load image {path}: {exc}")
    return images


def doc_to_audio(doc):
    audios = []
    for path in doc.get("audio", []):
        if not os.path.exists(path):
            print(f"Warning: Audio file not found: {path}")
            continue
        audio = load_audio_file(path)
        if audio is not None:
            audios.append(audio)
    return audios


def load_audio_file(file_path):
    try:
        import librosa

        audio_array, sampling_rate = librosa.load(file_path, sr=None)
        return {"array": audio_array, "sampling_rate": sampling_rate}
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Warning: Failed to load audio with librosa: {exc}")

    try:
        import numpy as np
        import soundfile as sf

        audio_array, sampling_rate = sf.read(file_path)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if len(audio_array.shape) > 1:
            audio_array = audio_array[:, 0]
        return {"array": audio_array, "sampling_rate": sampling_rate}
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Warning: Failed to load audio with soundfile: {exc}")

    return None
'''
    with open(multimodal_utils_dst, "w", encoding="utf-8") as f:
        f.write(fallback_content)
    print(f"Wrote fallback multimodal_utils.py to {multimodal_utils_dst}")
    return multimodal_utils_dst


def _normalise_remote_media_ref(ref_str: str) -> str:
    """Convert backend file URIs/absolute paths into bucket object names."""
    if ref_str.startswith("gs://"):
        parts = ref_str.removeprefix("gs://").split("/", 1)
        return parts[1] if len(parts) > 1 else parts[0]
    if "file:" in ref_str:
        ref_str = ref_str.split("file:", 1)[1]
    return ref_str.lstrip("/")


def copy_metrics_combined_to_temp(temp_dir: str = ".temp") -> str | None:
    """Write a proxy metrics_combined module to temp directory for lm_eval.

    lm_eval resolves !function references relative to the YAML directory and
    loads the module as a fresh instance. A plain copy would have
    CURRENT_COMBINED_FUNCTION=None since task.py sets it on the original
    module in sys.modules. This proxy delegates to the real module at call
    time so the global is resolved correctly.

    Args:
        temp_dir: Directory to write the proxy to

    Returns:
        Path to the proxy file, or None if source not found
    """
    metrics_src = "src/metrics_combined.py"
    metrics_dst = os.path.join(temp_dir, "src.metrics_combined.py")

    if os.path.exists(metrics_src):
        os.makedirs(temp_dir, exist_ok=True)
        proxy_content = (
            "import sys\n"
            "\n"
            "def _current_combined_process_results(doc, results):\n"
            "    real = sys.modules['src.metrics_combined']\n"
            "    return real._current_combined_process_results(doc, results)\n"
        )
        with open(metrics_dst, "w", encoding="utf-8") as f:
            f.write(proxy_content)
        print(f"Wrote metrics_combined proxy to {metrics_dst}")
        return metrics_dst
    print(f"Warning: {metrics_src} not found")
    return None


def _materialise_media(
    json_file_path: str,
    temp_dir: str,
    item_key: str,
    sub_dir: str,
    bucket: str | None,
) -> None:
    """Best-effort: pull every media reference into ``temp_dir/<sub_dir>``.

    Each item in the JSON list may contain a list under ``item_key`` (typically
    ``images`` / ``audio``). Each reference is resolved in this order:

    1. If it already exists on the local filesystem, copy it (local mode).
    2. Otherwise, if ``bucket`` is set, try downloading from
       ``gs://${bucket}/<reference>`` — pool files generated by the backend
       store relative GCS object paths.
    3. On any download failure, the reference is left untouched in the JSON
       and a warning is logged so the downstream lm_eval failure (FileNotFound)
       is easier to debug.
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    media_dir = os.path.join(temp_dir, sub_dir)
    os.makedirs(media_dir, exist_ok=True)

    # Lazy-init: avoid the import / auth cost when no media is needed.
    storage_client = None

    changed = False
    for item in data:
        refs = item.get(item_key)
        if not isinstance(refs, list):
            continue
        new_refs: list[str] = []
        for ref in refs:
            ref_str = str(ref)
            dst_path = os.path.join(media_dir, os.path.basename(ref_str))

            if os.path.exists(ref_str):
                if not os.path.exists(dst_path):
                    shutil.copy2(ref_str, dst_path)
                new_refs.append(dst_path)
                changed = True
                continue

            if bucket:
                object_ref = _normalise_remote_media_ref(ref_str)
                if storage_client is None:
                    try:
                        storage_client = storage.Client()
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(
                            f"[WARN] Could not init GCS client for media: {e}")
                        new_refs.append(ref_str)
                        continue
                try:
                    storage_client.bucket(bucket).blob(object_ref).download_to_filename(
                        dst_path
                    )
                    new_refs.append(dst_path)
                    changed = True
                    continue
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"[WARN] Could not fetch gs://{bucket}/{object_ref}: {e}")

            # Couldn't resolve — leave the reference as-is. Downstream lm_eval
            # will fail loudly with FileNotFound, which is more debuggable than
            # silently dropping the item.
            new_refs.append(ref_str)

        if changed:
            item[item_key] = new_refs

    if changed:
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def copy_images_to_temp(json_file_path: str, temp_dir: str, bucket: str | None = None) -> None:
    """Resolve image references for ``json_file_path`` into ``temp_dir/images``."""
    _materialise_media(json_file_path, temp_dir, "images", "images", bucket)


def copy_audio_to_temp(json_file_path: str, temp_dir: str, bucket: str | None = None) -> None:
    """Resolve audio references for ``json_file_path`` into ``temp_dir/audio``."""
    _materialise_media(json_file_path, temp_dir, "audio", "audio", bucket)


def set_api_key_for_adapter(adapter: str, api_key: str | None) -> None:
    """Set API key environment variable based on adapter type.

    Args:
        adapter: Adapter type (e.g., "openai-chat-completions")
        api_key: API key to set
    """
    if not api_key:
        return

    env_var_map = {
        "openai-chat-completions": "OPENAI_API_KEY",
        "local-chat-completions": "OPENAI_API_KEY",
        "anthropic-chat-completions": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "openai-asr": "OPENAI_API_KEY",
        "azure-stt": "AZURE_SPEECH_KEY",
    }

    env_var = env_var_map.get(adapter)
    if env_var:
        os.environ[env_var] = api_key
