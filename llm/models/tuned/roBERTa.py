import logging
from transformers import pipeline
import json
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(model_name, input_path, output_path):
    logger.info("Checking disk space...")
    utils.check_disk_space()

    logger.info("Loading the question-answering pipeline...")
    # Load the question-answering pipeline
    qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

    logger.info(f"Loading dataset from {input_path}...")
    # Load dataset data
    with open(input_path, "r") as f:
        dataset_data = json.load(f)

    logger.info("Generating answers for all questions in the dataset...")
    results = []

    # Process all questions
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

    logger.info(f"Saving the generated answers to {output_path}...")
    # Save the results
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)

    logger.info("Answers saved successfully.")


if __name__ == "__main__":
    model_name = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    main(model_name, input_path, output_path)
