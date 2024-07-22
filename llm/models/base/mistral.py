import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
import json
from tqdm import tqdm
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(model_name, input_path, output_path):

    tqdm.write("Loading the model and tokenizer...")
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    qa_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
    )

    tqdm.write(f"Loading dataset from {input_path}...")
    # Load dataset data
    with open(input_path, "r") as f:
        dataset_data = json.load(f)

    tqdm.write("Generating answers for all questions in the dataset...")
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
                prompt = f"<s>[INST] Context: {input_['context']} Question: {input_['question']} [/INST]"
                output = qa_pipeline(
                    prompt,
                    max_length=2048,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                answer = output[0]["generated_text"].split("[/INST]")[-1].strip()
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                answers.append("")

        best_answer = max(answers, key=len) if answers else ""
        return best_answer

    # Process all questions
    article_progress = tqdm(total=len(dataset_data["data"]), desc="Processing articles")
    for i, article in enumerate(dataset_data["data"]):
        question_progress = tqdm(
            total=sum(len(paragraph["qas"]) for paragraph in article["paragraphs"]),
            desc=f"Processing questions in article {i+1}",
        )
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                try:
                    answer = generate_answer(question, context)
                    results.append({"id": id_, "answer": answer})
                except Exception as e:
                    logger.error(f"Error generating answer for id {id_}: {e}")
                question_progress.update(1)
                question_progress.set_postfix(question=id_)
        question_progress.close()
        article_progress.update(1)
    article_progress.close()

    tqdm.write(f"Saving the generated answers to {output_path}...")
    # Save the results
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)

    tqdm.write("Answers saved successfully.")


if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    input_path = "../data/SQuAD/dev-v2.0.json"  # sys.argv[1]
    output_path = f"../model_results/base/{model_name}_predictions.json"  # sys.argv[2]
    main(model_name, input_path, output_path)
