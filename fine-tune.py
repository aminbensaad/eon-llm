import subprocess
import sys
import os
import shutil

# Install dependencies
# subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
# subprocess.check_call(
#     [
#         sys.executable,
#         "-m",
#         "pip",
#         "install",
#         "peft",
#         "bitsandbytes",
#         "datasets",
#         "transformers",
#         "scikit-learn",
#     ]
# )

import logging
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)
from datasets import DatasetDict, Dataset
import torch
import json
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, TaskType  # Import PEFT
from llm.scripts.utils.predict import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_disk_quota():
    """
    Print usage information about disk.
    This is for debugging purposes since models can require multiple GBs of space.
    """
    total, used, free = shutil.disk_usage("/")
    print(f"Total: {total // (2**30)} GB")
    print(f"Used: {used // (2**30)} GB")
    print(f"Free: {free // (2**30)} GB")


print_disk_quota()

# Define constants
base = "llm/"
MODEL_NAME = "deepset/roberta-large-squad2"  # Change to any model from the list
TRAIN_DATASET_PATH = os.path.join(base, "data/GermanQuAD/GermanQuAD_train.json")
OUTPUT_DIR = "./fine_tune_results"


def load_and_split_dataset(train_path, split_ratio=0.2):
    """
    Load data and generate a training and validation dataset.
    The training dataset can be used for the fine-tuning while the validation dataset
    is for evaluating the actual model performance in different epochs.

    :param str train_path: Path to JSON data to be loaded
    """
    train_data = load_dataset(train_path)

    articles = train_data["data"]
    contexts, questions, answers = [], [], []
    for article in articles:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                contexts.append(context)
                questions.append(qa["question"])
                answers.append(qa["answers"])

    (
        train_contexts,
        val_contexts,
        train_questions,
        val_questions,
        train_answers,
        val_answers,
    ) = train_test_split(
        contexts, questions, answers, test_size=split_ratio, random_state=42
    )

    train_dataset = Dataset.from_dict(
        {
            "context": train_contexts,
            "question": train_questions,
            "answers": train_answers,
        }
    )
    val_dataset = Dataset.from_dict(
        {"context": val_contexts, "question": val_questions, "answers": val_answers}
    )

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def print_model_layers(model):
    """
    Print names of all layers in given model to stdout.

    :param AutoModelForQuestionAnswering model: Model which should be inspected
    """
    for name, module in model.named_modules():
        print(name)


def check_target_modules(model, target_modules):
    """
    Check if model provides all given modules.

    :param AutoModelForQuestionAnswering model: Model of which models should be checked
    :param list target_modules: Modules which the given model should provide

    :return: Intersection of provided modules by model and desired target_modules
    """
    model_layers = [name for name, _ in model.named_modules()]
    missing_modules = [
        module for module in target_modules if module not in model_layers
    ]
    if missing_modules:
        logger.warning(
            f"Target modules {missing_modules} not found in the base model. Continuing without these modules."
        )
        return [module for module in target_modules if module not in missing_modules]
    return target_modules


def main():
    # Load the dataset
    logger.info("Loading the dataset...")
    datasets = load_and_split_dataset(TRAIN_DATASET_PATH)

    # Check dataset columns
    logger.info(f"Columns in the dataset: {datasets['train'].column_names}")

    # Load the tokenizer and model
    logger.info("Loading the tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Print model layers to identify the correct target modules
    print_model_layers(model)

    # Define target modules
    target_modules = ["query_key_value", "dense"]  # Adjust based on model architecture

    # Check if target modules exist in the model
    target_modules = check_target_modules(model, target_modules)

    # Apply QLoRA if target modules are found
    if target_modules:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Adjusting to TaskType.CAUSAL_LM for compatibility
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        model = get_peft_model(model, config)
    else:
        logger.info(
            "No valid target modules found for QLoRA. Proceeding without applying QLoRA."
        )

    # Preprocess the dataset
    logger.info("Preprocessing the dataset...")

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            if len(examples["answers"][i]) == 0:
                # No answer case
                start_positions.append(0)
                end_positions.append(0)
            else:
                answer = examples["answers"][i][
                    0
                ]  # Accessing the first answer in the list
                start_char = answer["answer_start"]
                end_char = start_char + len(answer["text"])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                context_start = sequence_ids.index(1)
                context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

                # If the answer is out of the context span, label as (0, 0)
                if not (
                    offsets[context_start][0] <= start_char
                    and offsets[context_end][1] >= end_char
                ):
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise, set the start and end positions to the character positions of the answer
                    start_positions.append(start_char - offsets[context_start][0])
                    end_positions.append(end_char - offsets[context_start][0])

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["context", "question", "answers"],
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",  # Ensure the save strategy matches the evaluation strategy
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Define the Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")

    # Save the model
    logger.info("Saving the model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
