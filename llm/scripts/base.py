import logging
from transformers import (
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
)
import torch
import json
from tqdm import tqdm
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO for standard output
logger = logging.getLogger(__name__)


def main(model_name, input_path, output_path):
    """
    :param str model_name: Model name from which it can be loaded with the HuggingFace
                           transformers API
    :param str input_path: Path to question-answering input data in JSON format
    :param str output_path: Path to which the answers should be written to in JSON format
    """

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    try:
        generation_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Loading dataset from {input_path}...")
    # Load dataset data
    with open(input_path, "r") as f:
        dataset_data = json.load(f)

    def answer_question_with_pipeline(question, context, max_new_tokens=250):
        """
        Generate answer for given context and question.
        """
        prompt = f"{question}\n{context}\nAnswer:"
        sequences = generation_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        return sequences[0]["generated_text"].split("Answer:")[1].strip()

    logger.info("Generating answers for all articles in the dev set...")
    results = []
    # Process all articles
    for i, article in enumerate(tqdm(dataset_data["data"], desc="Processing articles")):
        logger.info(f"Processing article {i + 1}/{len(dataset_data['data'])}")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                try:
                    answer = answer_question_with_pipeline(question, context)
                    results.append({"id": id_, "answer": answer})
                except Exception as e:
                    logger.error(f"Error generating answer for id {id_}: {e}")

    logger.info(f"Saving the generated answers to {output_path}...")
    # Save the results
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)

    logger.info("Answers saved successfully.")

    logger.info(f"Saving the generated answers to {output_path}...")
    # Save the results
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)

    logger.info("Answers saved successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <input_path> <output_path>")
        sys.exit(1)

    model_name = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    main(model_name, input_path, output_path)
