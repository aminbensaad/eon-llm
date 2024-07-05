import logging
import shutil
import os
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_disk_space(min_free_gb=30):
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    if free_gb < min_free_gb:
        logger.warning(f"Low disk space: {free_gb}GB available. Clearing cache.")
        clear_huggingface_cache()
    else:
        logger.info(f"Disk space is sufficient: {free_gb}GB available.")


def clear_huggingface_cache():
    cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Hugging Face cache cleared.")


def remove_safetensors_files(cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    # Find all safetensors files in the cache directory
    safetensors_files = glob.glob(
        os.path.join(cache_dir, "**/*.safetensors"), recursive=True
    )

    # Check if any safetensors files are found
    if not safetensors_files:
        print("No .safetensors files found.")
    else:
        # Delete each file
        for file_path in safetensors_files:
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")
