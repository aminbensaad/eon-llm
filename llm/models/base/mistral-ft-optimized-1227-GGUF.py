import sys
import os
import json
import logging
from llama_cpp import Llama
from tqdm import tqdm
import torch
from huggingface_hub import hf_hub_download

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(model_name, input_path, output_path):
    logger.info("Loading the model...")

    # Determine the number of CPU threads to use
    n_threads = os.cpu_count()
    logger.info(f"Number of CPU threads available: {n_threads}")

    # Check if GPU is available
    if torch.cuda.is_available():
        n_gpu_layers = 35  # adjust this value based on your GPU capability
        logger.info("GPU is available. Using GPU acceleration.")
    else:
        n_gpu_layers = 0
        logger.info("GPU is not available. Using CPU only.")

    # local model file path
    model_file_name = "mistral-ft-optimized-1227.Q4_K_M.gguf"
    local_model_path = os.path.join(".", model_file_name)

    # Check if the model file already exists locally
    if os.path.exists(local_model_path):
        logger.info(f"Model file already exists: {local_model_path}")
        model_path = local_model_path
    else:
        # Download model from Hugging Face
        try:
            model_path = hf_hub_download(repo_id=model_name, filename=model_file_name)
            logger.info(f"Model downloaded to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            sys.exit(1)

    # Set up the Llama model
    llm = Llama(
        model_path=model_path,  # Path to the GGUF model file
        n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=n_threads,  # Number of CPU threads to use
        n_gpu_layers=n_gpu_layers,  # Number of layers to offload to GPU, set to 0 if no GPU acceleration is available
    )

    logger.info(f"Loading dataset from {input_path}...")
    # Load dataset data
    with open(input_path, "r") as f:
        dataset_data = json.load(f)

    logger.info("Generating answers for all questions in the dataset...")
    results = []

    def generate_answer(question, context):
        prompt = f"[INST] Kontext: {context} Frage: {question} [/INST]"
        output = llm(
            prompt,  # Prompt
            max_tokens=512,  # Generate up to 512 tokens
            stop=[
                "</s>"
            ],  # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=True,  # Whether to echo the prompt
        )
        answer = output["choices"][0]["text"].strip()
        return answer

    # Process all questions
    for article in tqdm(dataset_data["data"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                if not qa.get("is_impossible", False):
                    answer = generate_answer(question, context)
                else:
                    answer = ""  # Handle unanswerable questions
                results.append({"id": id_, "answer": answer})

    logger.info(f"Saving the generated answers to {output_path}...")
    # Save the results
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)

    logger.info("Answers saved successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python mistral-ft-optimized-1227-GGUF.py <model_name> <input_path> <output_path>"
        )
        sys.exit(1)

    model_name = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    main(model_name, input_path, output_path)
