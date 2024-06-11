import logging
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys
import os
from accelerate import init_empty_weights, infer_auto_device_map
from accelerate.big_modeling import load_checkpoint_and_dispatch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(model_name, input_path, output_path):
    logger.info(f"Loading the tokenizer and model for {model_name}...")

    offload_folder = (
        f'./offload/offload_{model_name.split("/")[-1]}'  # Unique folder for each model
    )
    os.makedirs(offload_folder, exist_ok=True)

    # Load the tokenizer outside of init_empty_weights
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize and load the model with offloading
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_folder,
        )

    # Ensure weights are tied
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # Dispatch model properly with accelerate
    device_map = infer_auto_device_map(model, dtype=torch.float16)
    model = load_checkpoint_and_dispatch(
        model,
        offload_folder,
        device_map=device_map,
        offload_state_dict=True,
        offload_buffers=True,  # Handle offloading buffers
    )
    model.eval()

    logger.info("Loading input data...")
    # Load input data
    with open(input_path, "r") as f:
        squad_data = json.load(f)

    def generate_answer(question, context, max_length=512):
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_new_tokens=max_length)

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = (
            answer.split("Answer:")[1].strip()
            if "Answer:" in answer
            else answer.strip()
        )
        return answer

    logger.info("Generating answers for all articles in the input set...")
    results = []

    for article in tqdm(squad_data["data"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                answer = generate_answer(question, context)
                results.append({"id": id_, "answer": answer})

    logger.info(f"Saving the generated answers to {output_path}...")
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)

    logger.info("Answers saved successfully.")


if __name__ == "__main__":
    model_name = "tiiuae/falcon-7b"
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(model_name, input_path, output_path)
