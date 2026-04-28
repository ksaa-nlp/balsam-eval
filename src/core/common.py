"""Common utility functions shared across the project."""

import json
import os
import shutil
from pathlib import Path
from typing import Any


def setup_directories(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def copy_multimodal_utils_to_temp(temp_dir: str = ".temp") -> str | None:
    """Copy multimodal_utils.py to temp directory for lm_eval to find it.

    Args:
        temp_dir: Directory to copy the file to

    Returns:
        Path to the copied file, or None if source not found
    """
    multimodal_utils_src = "src/core/multimodal_utils.py"
    multimodal_utils_dst = os.path.join(temp_dir, "multimodal_utils.py")

    if os.path.exists(multimodal_utils_src):
        os.makedirs(temp_dir, exist_ok=True)
        shutil.copy2(multimodal_utils_src, multimodal_utils_dst)
        print(f"Copied multimodal_utils.py to {multimodal_utils_dst}")
        return multimodal_utils_dst
    else:
        print(f"Warning: {multimodal_utils_src} not found")
        return None


def copy_images_to_temp(json_file_path: str, temp_dir: str) -> None:
    """Copy images referenced in JSON file to temp directory.

    Args:
        json_file_path: Path to JSON file containing image references
        temp_dir: Directory to copy images to
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for item in data:
        if 'images' in item and isinstance(item['images'], list):
            for img_path in item['images']:
                if os.path.exists(img_path):
                    img_name = os.path.basename(img_path)
                    dst_path = os.path.join(images_dir, img_name)
                    if not os.path.exists(dst_path):
                        shutil.copy2(img_path, dst_path)
                        print(f"Copied image: {img_name} -> {dst_path}")
                    # Update path in item to relative path
                    item['images'] = [dst_path if p == img_path else p for p in item['images']]

    # Write back updated JSON
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def setup_environment() -> dict[str, str]:
    """Load and return environment variables.

    Returns:
        Dictionary of environment variables

    Raises:
        ValueError: If required environment variables are missing
    """
    from dotenv import load_dotenv
    load_dotenv()

    env_vars = {
        'BASE_URL': os.getenv("BASE_URL"),
        'API_KEY': os.getenv("API_KEY"),
        'MODEL_NAME': os.getenv("MODEL"),
        'ADAPTER': os.getenv("ADAPTER"),
        'SERVER_TOKEN': os.getenv("SERVER_TOKEN"),
        'API_HOST': os.getenv("API_HOST"),
        'USER_ID': os.getenv("USER_ID"),
        'BENCHMARK_ID': os.getenv("BENCHMARK_ID"),
        'EVALUATION_TYPES': os.getenv("EVALUATION_TYPES"),
        'LLM_JUDGE': os.getenv("JUDGE_MODEL"),
        'LLM_JUDGE_PROVIDER': os.getenv("JUDGE_PROVIDER"),
        'LLM_JUDGE_API_KEY': os.getenv("JUDGE_API_KEY"),
        'CATEGORY_ID': os.getenv("CATEGORY"),
        'JOB_ID': os.getenv("JOB_ID"),
        'TEMPERATURE': os.getenv("TEMPERATURE"),
    }

    return env_vars


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
        "humain": "HUMAIN_API_KEY",
    }

    env_var = env_var_map.get(adapter)
    if env_var:
        os.environ[env_var] = api_key
