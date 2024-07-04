import os
import time
import logging
import argparse
import sys

# Ensure the script is running in the "llm/scripts" directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
base = ".."
data_dir = os.path.join(base, "data")
model_results_dir = os.path.join(base, "model_results")
eval_results_dir = os.path.join(base, "eval_results")
metrics_dir = os.path.join(base, "metrics")
model_dir = os.path.join(base, "models")
timing_results_path = os.path.join(base, "timing_results.json")

# Define models to run
model_IDs = {
    "base": [  # German base models (pre-trained)
        #"TheBloke/mistral-ft-optimized-1227-GGUF",
         "tiiuae/falcon-7b-instruct", # ✅
        # "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
         "nvidia/Llama3-ChatQA-1.5-8B",  # ✅
        # "mistralai/Mistral-7B-Instruct-v0.1",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        # "mistralai/Mixtral-8x7B-Instruct-v0.1",  # PENDING access
        # "Deci/DeciLM-7B-instruct",
        # HuggingFace Leaderboard
        # "BarraHome/Mistroll-7B-v2.2",
        # "yam-peleg/Experiment26-7B",
        # "MTSAIR/multi_verse_model",
    ],
    "Gbase": [  # German base models (pre-trained)
        # "philschmid/instruct-igel-001",
        # "TheBloke/em_german_leo_mistral-GGUF",
        "VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct",
        # "TheBloke/DiscoLM_German_7b_v1-GGUF",
    ],
    "tuned": [  # General models (fine-tuned on SQuAD)
         "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",  # ✅
        "distilbert/distilbert-base-cased-distilled-squad",  # ✅
         "timpal0l/mdeberta-v3-base-squad2",  # ✅
         "deepset/roberta-base-squad2",  # ✅
         "deepset/roberta-large-squad2",  # ✅
         #"deepset/xlm-roberta-base-squad2",
    ],
    "Gtuned": [  # German models (fine-tuned on GermanQuAD)
        "deutsche-telekom/bert-multi-english-german-squad2",  # ✅
        "deepset/gelectra-base-germanquad-distilled",
        "deepset/gelectra-base-germanquad",  # ✅
        "deepset/gelectra-large-germanquad",  # ✅
    ],
}


def model_name_from_id(model_id: str):
    model_id_components = model_id.split("/")
    if len(model_id_components) > 2:
        return model_id_components[-2]
    else:
        return model_id_components[-1]


# Define dataset paths
datasets = {
    "SQuAD": os.path.join(data_dir, "SQuAD/dev-v2.0.json"),
    "G": os.path.join(data_dir, "GermanQuAD/GermanQuAD_test.json"),
}


def print_usage():
    print("Usage: python run.py [options] -d <dataset>... -m <model_type>...")
    print("Options:")
    print("  -d, --datasets        Specify the datasets to use (SQuAD, G).")
    print(
        "  -m, --model_types     Specify the model types to run (base, Gbase, tuned, Gtuned)."
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
        choices=["base", "Gbase", "tuned", "Gtuned"],
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
                    model_name = model_ID.split("/")[1]
                    output_file_name = f"{model_name}{suffix}_predictions.json"
                    output_path = os.path.join(
                        model_results_dir, model_type, output_file_name
                    )
                    print(f"Output path is {output_path}")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    script = os.path.join(model_dir, model_type, f"{model_name}.py")

                    logger.info("Checking disk space...")
                    utils.check_disk_space()
                    start_time = time.time()
                    utils.run_model_script(script, model_ID, input_path, output_path)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    # Store timing results
                    utils.store_timing_results(
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
                            model_name = model_ID.split("/")[1]
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
                                utils.evaluate_model_results(
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
