import os
import subprocess
import time
import logging
import argparse
import json

# Ensure the script is running in the "llm/scripts" directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
base = ".."
data_dir = os.path.join(base, "data")
model_results_dir = os.path.join(base, "model_results")
eval_results_dir = os.path.join(base, "eval_results")
metrics_dir = os.path.join(base, "metrics")
eval_script = os.path.join(metrics_dir, "evaluate-v2.0.py")
bleu_script = os.path.join(metrics_dir, "bleu.py")
rouge_script = os.path.join(metrics_dir, "rouge.py")


# Define models to run
model_scripts = {
    "base": [  # German base models (pre-trained)
        # Community Suggestions
        "tiiuae/falcon-7b",
        "meta-llama/Meta-Llama-3-8B",  # PENDING access
        "mistralai/Mixtral-8x7B-v0.1",  # PENDING access
        "Deci/DeciLM-7B",
        # HuggingFace Leaderboard
        "BarraHome/Mistroll-7B-v2.2",
        "yam-peleg/Experiment26-7B",
        "MTSAIR/multi_verse_model",
    ],
    "Gbase": [  # German base models (pre-trained)
        "philschmid/instruct-igel-001",
        "TheBloke/em_german_leo_mistral-GGUF",
        "VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct",
    ],
    "tuned": [  # General models (fine-tuned on SQuAD)
        "timpal0l/mdeberta-v3-base-squad2",  # ✅
        "distilbert/distilbert-base-cased-distilled-squad",  # ✅
        "deepset/roberta-base-squad2",
        "deepset/roberta-large-squad2",
        "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",  # ✅
    ],
    "Gtuned": [  # German models (fine-tuned on GermanQuAD)
        "deepset/gelectra-base-germanquad",  # ✅
        "deepset/gelectra-large-germanquad",
        "deutsche-telekom/bert-multi-english-german-squad2",  # ✅
    ],
}

# Define dataset paths
squad_data_path = os.path.join(data_dir, "SQuAD/dev-v2.0.json")
germanquad_data_path = os.path.join(data_dir, "GermanQuAD/GermanQuAD_test.json")


def print_usage():
    print(
        "Usage: python run.py <model_type> [dataset] [-p] [-e] [--bleu] [--rouge] [--all]"
    )
    print("Options:")
    print(
        "  <model_type>          Specify the model type to run (base, Gbase, tuned, Gtuned)."
    )
    print(
        "  [dataset]             Specify the dataset to use (default: SQuAD, optional: G for GermanQuAD)."
    )
    print("  -p, --predictions     Run predictions.")
    print("  -e, --evaluations     Run evaluations.")
    print("  --bleu                Run BLEU metric evaluation.")
    print("  --rouge               Run ROUGE metric evaluation.")
    print("  --all                 Run all metrics evaluation.")
    print("By default, only evaluations are run if no options are provided.\n")
    print("===================================================================\n\n")


# Function to run a model script
def run_model_script(model_name, model_type, input_path, output_path):
    script_name = "tuned.py" if "tuned" in model_type else "base.py"
    command = ["python", script_name, model_name, input_path, output_path]

    logger.info(f"Running {script_name} for {model_name}...")
    subprocess.run(command)
    logger.info(f"{script_name} for {model_name} completed.")


# Function to evaluate the model results
def evaluate_model_results(model_name, model_type, dataset, metrics):
    predictions_path = os.path.join(
        model_results_dir, model_type, f"{model_name}_predictions.json"
    )
    eval_output_path = os.path.join(
        eval_results_dir, model_type, f"{model_name}_eval_results.json"
    )

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

    # Evaluate using evaluate-v2.0.py
    if "evaluate-v2" in metrics:
        command = [
            "python",
            eval_script,
            squad_data_path if dataset == "SQuAD" else germanquad_data_path,
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

    # Evaluate BLEU
    if "bleu" in metrics:
        bleu_output_path = os.path.join(
            eval_results_dir, model_type, f"{model_name}_bleu_results.json"
        )
        command_bleu = [
            "python",
            bleu_script,
            predictions_path,
            squad_data_path if dataset == "SQuAD" else germanquad_data_path,
            bleu_output_path,
        ]
        logger.info(f"Evaluating BLEU for {model_name}...")
        subprocess.run(command_bleu)
        logger.info(f"BLEU evaluation for {model_name} completed.")

        with open(bleu_output_path, "r") as f:
            bleu_results = json.load(f)
            results["bleu"] = bleu_results

        # Remove temporary BLEU results file
        os.remove(bleu_output_path)

    # Evaluate ROUGE
    if "rouge" in metrics:
        rouge_output_path = os.path.join(
            eval_results_dir, model_type, f"{model_name}_rouge_results.json"
        )
        command_rouge = [
            "python",
            rouge_script,
            predictions_path,
            squad_data_path if dataset == "SQuAD" else germanquad_data_path,
            rouge_output_path,
        ]
        logger.info(f"Evaluating ROUGE for {model_name}...")
        subprocess.run(command_rouge)
        logger.info(f"ROUGE evaluation for {model_name} completed.")

        with open(rouge_output_path, "r") as f:
            rouge_results = json.load(f)
            results["rouge"] = rouge_results

        # Remove temporary ROUGE results file
        os.remove(rouge_output_path)

    # Save combined results
    with open(eval_output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions and/or evaluations.")
    parser.add_argument(
        "model_type",
        type=str,
        choices=["base", "Gbase", "tuned", "Gtuned"],
        help="Specify the model type to run.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="SQuAD",
        choices=["SQuAD", "G"],
        help="Specify the dataset to use (default: SQuAD, optional: G for GermanQuAD).",
    )
    parser.add_argument(
        "-p", "--predictions", action="store_true", help="Run predictions."
    )
    parser.add_argument(
        "-e", "--evaluations", action="store_true", help="Run evaluations."
    )
    parser.add_argument(
        "--bleu", action="store_true", help="Run BLEU metric evaluation."
    )
    parser.add_argument(
        "--rouge", action="store_true", help="Run ROUGE metric evaluation."
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all metrics evaluation."
    )

    args = parser.parse_args()

    if not args.predictions and not args.evaluations:
        args.evaluations = True

    print_usage()

    metrics_to_run = []
    if args.all:
        metrics_to_run = ["evaluate-v2", "bleu", "rouge"]
    else:
        if args.evaluations:
            metrics_to_run.append("evaluate-v2")
        if args.bleu:
            metrics_to_run.append("bleu")
        if args.rouge:
            metrics_to_run.append("rouge")

    # Determine the dataset to use
    if args.dataset == "G" or "G" in args.model_type:
        dataset = "GermanQuAD"
        input_path = germanquad_data_path
    else:
        dataset = "SQuAD"
        input_path = squad_data_path

    model_type = args.model_type
    models = model_scripts[model_type]

    # Run predictions if specified
    if args.predictions:
        for model_script in models:
            model_name = model_script.split("/")[1]
            output_path = os.path.join(
                model_results_dir, model_type, f"{model_name}_predictions.json"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            run_model_script(model_script, model_type, input_path, output_path)

    # Wait for results to appear in model_results (if necessary)
    if args.predictions:
        time.sleep(5)  # Adjust the sleep time if needed

    # Run evaluations if specified
    if args.evaluations or args.bleu or args.rouge:
        for model_script in models:
            model_name = model_script.split("/")[1]
            try:
                evaluate_model_results(model_name, model_type, dataset, metrics_to_run)
            except Exception as e:
                logger.warning(f"Failed to evaluate model {model_name}: {e}")
