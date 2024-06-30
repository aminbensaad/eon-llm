import logging
import shutil
import os
import json
import matplotlib.pyplot as plt
import subprocess
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

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
    dataset_file_path,
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

    # Evaluate BLEU
    if "bleu" in metrics:
        evaluate_metric(bleu_script, "bleu")

    # Evaluate ROUGE
    if "rouge" in metrics:
        evaluate_metric(rouge_script, "rouge")

    # Evaluate BERTScore
    if "bertscore" in metrics:
        extra_args = ["-G"] if dataset == "G" else None
        evaluate_metric(bertscore_script, "bertscore", extra_args)

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


logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name):
    logger.info("Loading the tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def load_pipeline(model_name):
    logger.info("Loading the question-answering pipeline...")
    qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return qa_pipeline


def load_dataset(input_path):
    logger.info(f"Loading dataset from {input_path}...")
    with open(input_path, "r") as f:
        dataset_data = json.load(f)
    return dataset_data


def save_results(output_path, results):
    logger.info(f"Saving the generated answers to {output_path}...")
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)
    logger.info("Answers saved successfully.")


def answer_question_with_sliding_window(
    tokenizer,
    model,
    device,
    question,
    context,
    use_token_type_ids=True,
    max_length=512,
    stride=128,
):
    inputs = tokenizer(
        question,
        context,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        stride=stride,
        return_overflowing_tokens=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    if use_token_type_ids:
        token_type_ids = inputs["token_type_ids"].to(device)
    else:
        token_type_ids = None

    all_answers = []
    with torch.no_grad():
        if token_type_ids is not None:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    for i in range(input_ids.size(0)):
        start_scores = start_logits[i]
        end_scores = end_logits[i]
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[i][answer_start:answer_end])
        )
        all_answers.append(answer)

    best_answer = max(all_answers, key=len)
    return best_answer


def generate_answers_with_sliding_window(
    model_name, input_path, output_path, use_token_type_ids=True, max_length=512
):
    tokenizer, model, device = load_model_and_tokenizer(model_name)
    dataset_data = load_dataset(input_path)
    results = []

    logger.info("Generating answers for all articles in the dataset...")
    for article in tqdm(dataset_data["data"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                answer = answer_question_with_sliding_window(
                    tokenizer,
                    model,
                    device,
                    question,
                    context,
                    use_token_type_ids,
                    max_length,
                )
                results.append({"id": id_, "answer": answer})

    save_results(output_path, results)


def generate_answers_with_pipeline(model_name, input_path, output_path):
    qa_pipeline = load_pipeline(model_name)
    dataset_data = load_dataset(input_path)
    results = []

    logger.info("Generating answers for all questions in the dataset...")
    for article in tqdm(dataset_data["data"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                qa_input = {"question": question, "context": context}
                try:
                    answer = qa_pipeline(qa_input)["answer"]
                except Exception as e:
                    logger.error(f"Error processing question ID {id_}: {e}")
                    answer = ""

                results.append({"id": id_, "answer": answer})

    save_results(output_path, results)
