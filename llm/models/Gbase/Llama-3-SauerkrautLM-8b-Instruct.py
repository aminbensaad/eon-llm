import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from tqdm import tqdm
import torch
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(model_name, input_path, output_path):

    logger.info("Loading the text-generation pipeline...")
    # Load the text-generation pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    logger.info(f"Loading dataset from {input_path}...")
    # Load dataset data
    with open(input_path, "r") as f:
        dataset_data = json.load(f)

    logger.info("Generating answers for all questions in the dataset...")
    results = []

    max_length = 512  # Max length for the model
    doc_stride = 128  # Stride size for overlapping contexts

    def generate_answer(question, context):
        inputs = []
        start = 0
        while start < len(context):
            end = min(start + max_length - len(question.split()) - 3, len(context))
            inputs.append({"question": question, "context": context[start:end]})
            if end == len(context):
                break
            start += doc_stride

        answers = []
        for input_ in inputs:
            try:
                prompt = f"system\n\nDu bist ein freundlicher und hilfreicher deutscher KI-Assistent.\nuser\n\n{question}\nassistant\n\n"
                output = qa_pipeline(
                    prompt,
                    max_length=2048,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                )
                answer = output[0]["generated_text"].split("assistant\n\n")[-1].strip()
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                answers.append("")

        best_answer = max(answers, key=len) if answers else ""
        return best_answer

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
            "Usage: python llama-3-sauerkrautlm-8b-instruct.py <model_name> <input_path> <output_path>"
        )
        sys.exit(1)

    model_name = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    main(model_name, input_path, output_path)
