import logging
import shutil
import os
import json
import matplotlib.pyplot as plt
import subprocess


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


# Function to run a model script
def run_model_script(script, model_ID, input_path, output_path):
    if os.path.exists(script):
        command = ["python", script, model_ID, input_path, output_path]
        logger.info(f"Running {script} for {model_ID}...")
        try:
            subprocess.run(command, check=True)
            logger.info(f"{script} for {model_ID} completed.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {script} for {model_ID}: {e}")
    else:
        logger.error(f"Script {script} does not exist. Skipping {model_ID}.")


# Function to evaluate the model results
def evaluate_model_results(
    metrics_dir,
    eval_results_dir,
    predictions_path,
    eval_output_path,
    model_name,
    model_type,
    dataset,
    metrics,
):
    # Scripts
    eval_script = os.path.join(metrics_dir, "evaluate-v2.0.py")
    bleu_script = os.path.join(metrics_dir, "bleu.py")
    rouge_script = os.path.join(metrics_dir, "rouge.py")
    bertscore_script = os.path.join(metrics_dir, "BERT-score.py")

    if not os.path.exists(predictions_path):
        logger.warning(
            f"Predictions file for {model_name} not found. Skipping evaluation."
        )
        return

    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)

    # Load existing results if they exist
    if os.path.exists(eval_output_path):
        with open(eval_output_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    isGermanQuAD = "German" in dataset

    def save_results():
        with open(eval_output_path, "w") as f:
            json.dump(results, f, indent=2)

    def evaluate_metric(script, metric_name, output_file_suffix, extra_args=None):
        output_path = os.path.join(
            eval_results_dir,
            model_type,
            f"{model_name}_{output_file_suffix}_results.json",
        )
        command = ["python", script, predictions_path, dataset, output_path]
        if extra_args:
            command.extend(extra_args)
        logger.info(f"Evaluating {metric_name} for {model_name}...")
        subprocess.run(command)
        logger.info(f"{metric_name} evaluation for {model_name} completed.")

        with open(output_path, "r") as f:
            metric_results = json.load(f)
            results[metric_name] = metric_results

        # Save results after each metric evaluation
        save_results()

        # Remove temporary results file
        os.remove(output_path)

    # Evaluate using evaluate-v2.0.py
    if "evaluate-v2" in metrics:
        command = [
            "python",
            eval_script,
            dataset,
            predictions_path,
            "--out-file",
            eval_output_path,
            "--na-prob-thresh",
            "0.5",
        ]
        logger.info(f"Evaluating evaluate-v2 for {model_name}...")
        subprocess.run(command)
        logger.info(f"Evaluate-v2 evaluation for {model_name} completed.")

        with open(eval_output_path, "r") as f:
            eval_results = json.load(f)
            results["evaluate-v2"] = eval_results

        save_results()

    # Evaluate BLEU
    if "bleu" in metrics:
        evaluate_metric(bleu_script, "bleu", "bleu")

    # Evaluate ROUGE
    if "rouge" in metrics:
        evaluate_metric(rouge_script, "rouge", "rouge")

    # Evaluate BERTScore
    if "bertscore" in metrics:
        extra_args = ["-G"] if isGermanQuAD else None
        evaluate_metric(bertscore_script, "bertscore", "bertscore", extra_args)

    # Save combined results at the end just in case
    save_results()


def store_timing_results(timing_results_path, model_name, dataset, elapsed_time):
    # Initialize timing results dictionary
    timing_results = {}
    if os.path.exists(timing_results_path):
        with open(timing_results_path, "r") as f:
            timing_results = json.load(f)

    # Store timing results
    if model_name not in timing_results:
        timing_results[model_name] = {}
    timing_results[model_name][dataset] = elapsed_time

    # Save timing results
    with open(timing_results_path, "w") as f:
        json.dump(timing_results, f, indent=4)
