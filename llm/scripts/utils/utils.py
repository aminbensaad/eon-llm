import logging
import shutil
import os
import json
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_disk_space(min_free_gb=1):
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


def plot_answer_length_distribution(base_dir):
    all_answer_lengths = []

    # Iterate through all JSON files in the directory
    for file_name in os.listdir(base_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(base_dir, file_name)
            with open(file_path, "r") as file:
                data = json.load(file)
                # Extract answer lengths
                answer_lengths = [
                    len(str(answer))
                    for answer in data.values()
                    if isinstance(answer, (str, int, float))
                ]
                all_answer_lengths.extend(answer_lengths)

    # Plot the distribution of answer lengths
    plt.figure(figsize=(10, 6))
    plt.hist(all_answer_lengths, bins=30, edgecolor="k", alpha=0.7)
    plt.xlabel("Answer Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Answer Lengths Across All Files")
    plt.grid(True)
    plt.show()
