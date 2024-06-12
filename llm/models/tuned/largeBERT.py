import logging
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(input_path, output_path):
    logger.info("Loading the tokenizer and model...")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Loading SQuAD dev data...")
    # Load SQuAD dev data
    with open(input_path, "r") as f:
        squad_data = json.load(f)

    def answer_question_with_sliding_window(
        question, context, max_length=512, stride=128
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
            return_token_type_ids=True,  # BERT uses token type IDs
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)

        all_answers = []
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
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

    logger.info("Generating answers for all articles in the dev set...")
    results = []

    # Process all articles
    for article in tqdm(squad_data["data"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                answer = answer_question_with_sliding_window(question, context)
                results.append({"id": id_, "answer": answer})

    logger.info(f"Saving the generated answers to {output_path}...")
    # Save the results
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)

    logger.info("Answers saved successfully.")


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)
