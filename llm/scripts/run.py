import os
import time
import logging
import argparse
import sys

# Ensure the script is running in the "llm/scripts" directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "scripts")))

import utils.run_utils as run
import utils.core as core

from model_ids import (
    base_path,
    model_script_path,
    local_model_dir,
    model_IDs,
    model_name_from_id,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
data_dir = os.path.join(base_path, "data")
model_results_dir = os.path.join(base_path, "model_results")
eval_results_dir = os.path.join(base_path, "eval_results")
metrics_dir = os.path.join(base_path, "metrics")
timing_results_path = os.path.join(base_path, "timing_results.json")


# Define dataset paths
datasets = {
    "SQuAD": os.path.join(data_dir, "SQuAD/dev-v2.0.json"),
    "G": os.path.join(data_dir, "GermanQuAD/GermanQuAD_test.json"),
}


def print_usage():
    """
    Print usage text to stdout.
    """
    print("Usage: python run.py [options] -d <dataset>... -m <model_type>...")
    print("Options:")
    print("  -d, --datasets        Specify the datasets to use (SQuAD, G).")
    print(
        "  -m, --model_types     Specify the model types to run (base, Gbase, tuned, Gtuned, local)."
    )
    print("  -p, --predictions     Run predictions.")
    print("  -e, --evaluations     Run evaluations.")
    print("  --bleu                Run BLEU metric evaluation.")
    print("  --rouge               Run ROUGE metric evaluation.")
    print("  --bertscore           Run BERTScore metric evaluation.")
    print("  --all                 Run all metrics evaluation.")
    print("By default, only evaluations are run if no options are provided.\n")
    print("===================================================================\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions and/or evaluations.")
    parser.add_argument(
        "-m",
        "--model_types",
        type=str,
        nargs="+",
        choices=["base", "Gbase", "tuned", "Gtuned", "local"],
        help="Specify the model types to run.",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        nargs="+",
        choices=["SQuAD", "G"],
        help="Specify the datasets to use (SQuAD, G).",
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
        "--bertscore", action="store_true", help="Run BERTScore metric evaluation."
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
        metrics_to_run = ["evaluate-v2", "bleu", "rouge", "bertscore", "overall"]
    else:
        if args.evaluations:
            metrics_to_run.append("evaluate-v2")
        if args.bleu:
            metrics_to_run.append("bleu")
        if args.rouge:
            metrics_to_run.append("rouge")
        if args.bertscore:
            metrics_to_run.append("bertscore")

    for dataset_key in args.datasets:
        dataset = dataset_key
        input_path = datasets[dataset_key]
        suffix = "_G" if dataset_key == "G" else ""
        print(f"Input path is {input_path}")

        for model_type in args.model_types:
            models = model_IDs[model_type]

            # Run predictions if specified
            if args.predictions:
                for model_ID in models:
                    model_name = model_name_from_id(model_ID, model_type)
                    output_file_name = f"{model_name}{suffix}_predictions.json"
                    output_path = os.path.join(
                        model_results_dir, model_type, output_file_name
                    )
                    print(f"Output path is {output_path}")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    script = model_script_path(model_type, model_ID)

                    if model_type == "local":
                        model_ID = os.path.join(local_model_dir, model_ID)

                    logger.info("Checking disk space...")
                    core.check_disk_space()
                    start_time = time.time()
                    run.run_model_script(script, model_ID, input_path, output_path)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    # Store timing results
                    run.store_timing_results(
                        timing_results_path, model_name, dataset, elapsed_time
                    )

            # Wait for results to appear in model_results (if necessary)
            if args.predictions:
                time.sleep(5)  # Adjust the sleep time if needed

            # Run evaluations if specified
            if args.evaluations or args.bleu or args.rouge or args.bertscore:
                for model_type in args.model_types:
                    for dataset_key in args.datasets:
                        dataset = dataset_key
                        input_path = datasets[dataset_key]
                        suffix = "_G" if dataset_key == "G" else ""
                        models = model_IDs[model_type]

                        for model_ID in models:
                            model_name = model_name_from_id(model_ID, model_type)
                            try:
                                predictions_path = os.path.join(
                                    model_results_dir,
                                    model_type,
                                    f"{model_name}{suffix}_predictions.json",
                                )
                                eval_output_path = os.path.join(
                                    eval_results_dir,
                                    model_type,
                                    f"{model_name}_eval_results.json",
                                )
                                run.evaluate_model_results(
                                    metrics_dir,
                                    eval_results_dir,
                                    predictions_path,
                                    eval_output_path,
                                    model_name,
                                    model_type,
                                    dataset,
                                    input_path,
                                    metrics_to_run,
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to evaluate model {model_name}: {e}"
                                )
