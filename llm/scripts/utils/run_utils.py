import os
import json
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to run a model script
def run_model_script(script, model_ID, input_path, output_path):
    """
    Execute evaluation with given script for the model.

    :param str script: Path to inference script of model
    :param str model_ID: Name of model with which it can be loaded via HuggingFace
    :param str input_path: Path to JSON file with input dataset
    :param str output_path: Path to which the generated results should be written to
    """
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


# Function to evaluate the model results
def evaluate_model_results(
    metrics_dir,
    eval_results_dir,
    predictions_path,
    eval_output_path,
    model_name,
    model_type,
    dataset,
    dataset_file_path,
    metrics,
):
    """
    Run evaluation scripts on given model.

    :param str metrics_dir: Path to directory containing scripts to caluclate metrics
    :param str eval_results_dir: Path to directory to which intermediate evaluation results will be written
    :param str predictions_path: Path to JSON files containing the inference results
    :param str eval_output_path: Output path to which the evaluation result should be written to
    :param str model_name: Name of model which should be evaluated
    :param str model_type: Category of the model (e.g. "base", "tuned", ...)
    :param str dataset: Name of dataset ("SQuAD" or "G" (for GermanQuAD))
    :param str dataset_file_path: Path to dataset which was used for the predictions
    :param list[str] metrics: Metrics which should be caluclated
    """
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

    if dataset not in results:
        results[dataset] = {}

    def save_results():
        with open(eval_output_path, "w") as f:
            json.dump(results, f, indent=2)

    def evaluate_metric(script, metric_name, extra_args=None):
        output_path = os.path.join(
            eval_results_dir,
            model_type,
            f"{model_name}_{dataset}_{metric_name}_results.json",
        )
        command = ["python", script, predictions_path, dataset_file_path, output_path]
        if extra_args:
            command.extend(extra_args)
        logger.info(
            f"Evaluating {metric_name} for {model_name} on {dataset} dataset..."
        )
        subprocess.run(command)
        logger.info(
            f"{metric_name} evaluation for {model_name} on {dataset} dataset completed."
        )

        if not os.path.exists(output_path):
            logger.warning(f"Evaluation result for {metric_name} not found. Skipping.")
            return

        with open(output_path, "r") as f:
            metric_results = json.load(f)
            results[dataset][metric_name] = metric_results

        # Save results after each metric evaluation
        save_results()
        # Remove temporary results file
        os.remove(output_path)

    # Evaluate using evaluate-v2.0.py
    if "evaluate-v2" in metrics:
        output_path = os.path.join(
            eval_results_dir,
            model_type,
            f"{model_name}_{dataset}_evaluate-v2_results.json",
        )
        command = [
            "python",
            eval_script,
            dataset_file_path,
            predictions_path,
            "--out-file",
            output_path,
            "--na-prob-thresh",
            "0.5",
        ]
        logger.info(f"Evaluating evaluate-v2 for {model_name} on {dataset} dataset...")
        subprocess.run(command)
        logger.info(
            f"Evaluate-v2 evaluation for {model_name} on {dataset} dataset completed."
        )

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                eval_results = json.load(f)
                results[dataset]["evaluate-v2"] = eval_results

            save_results()
            os.remove(output_path)

    # Evaluate BLEU
    bleu = "bleu"
    if bleu in metrics:
        evaluate_metric(bleu_script, bleu)

    # Evaluate ROUGE
    rouge = "rouge"
    if rouge in metrics:
        evaluate_metric(rouge_script, rouge)

    # Evaluate BERTScore
    bertscore = "bertscore"
    if bertscore in metrics:
        extra_args = ["-G"] if dataset == "G" else None
        evaluate_metric(bertscore_script, bertscore, extra_args)

    overall = "overall"
    if overall in metrics:
        exact_score = results[dataset]["evaluate-v2"]["exact"]
        f1_score = results[dataset]["evaluate-v2"]["f1"]
        bleu_score = results[dataset]["bleu"]["bleu"]
        rouge_score = results[dataset]["rouge"]["rouge"]["rougeL"]["f"]
        bertscore_score = results[dataset]["bertscore"]["BERTScore"]["F1"]

        evalv2_score = (exact_score + f1_score) / 2.0
        bbr = (bertscore_score + bleu_score + rouge_score) / 3.0
        results[dataset][overall] = (evalv2_score + bbr) / 2.0

    # Save combined results at the end just in case
    save_results()
