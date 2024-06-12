import os
import glob

# Define the Hugging Face cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

# Find all safetensors files in the cache directory
safetensors_files = glob.glob(
    os.path.join(cache_dir, "**/*.safetensors"), recursive=True
)

# Delete each file
for file_path in safetensors_files:
    try:
        os.remove(file_path)
        print(f"Removed: {file_path}")
    except OSError as e:
        print(f"Error removing {file_path}: {e}")
