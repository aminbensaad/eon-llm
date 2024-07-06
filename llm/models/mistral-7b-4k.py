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
from awq import AutoAWQForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm.scripts.utils import predict

# Set up logging
logging.basicConfig(level=logging.INFO)  # Changed to INFO for standard output
logger = logging.getLogger(__name__)

# Quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
# Falcon Pipeline
model_id = "TheBloke/Mistral-7B-v0.1-AWQ"
model = AutoAWQForCausalLM.from_quantized(
    model_id, fuse_layers=True, trust_remote_code=False, safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)


def main(input_path, output_path):

    # Load the tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # try:
    # generation_pipeline = pipeline(
    #     "text-generation",
    #     model=model_name,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )
    # except Exception as e:
    # logger.error(f"Failed to load model: {e}")
    # return

    logger.info(f"Loading dataset from {input_path}...")
    # Load dataset data
    with open(input_path, "r") as f:
        dataset_data = json.load(f)

    def answer_question_with_pipeline(question, context, max_new_tokens=250):
        prompt = f"{question}\n{context}\nAnswer:"

        tokens = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        generation_output = model.generate(
            tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_new_tokens=512,
        )
        print(
            tokenizer.decode(
                generation_output[0]["generated_text"].split("Answer:")[1]
            ).strip()
        )
        return tokenizer.decode(
            generation_output[0]["generated_text"].split("Answer:")[1]
        ).strip()

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
    if len(sys.argv) != 3:
        print("Usage: python script.py input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)
