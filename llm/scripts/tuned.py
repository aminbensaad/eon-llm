import logging
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(model_name, input_path, output_path):
    """
    :param str model_name: Model name from which it can be loaded with the HuggingFace
                           transformers API
    :param str input_path: Path to question-answering input data in JSON format
    :param str output_path: Path to which the answers should be written to in JSON format
    """

    logger.info("Loading the tokenizer and model...")
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract max_length from the model's config
    max_length = model.config.max_position_embeddings

    # Determine if token type IDs are required
    return_token_type_ids = (
        hasattr(model.config, "type_vocab_size") and model.config.type_vocab_size > 1
    )

    logger.info(f"Loading dataset from {input_path}...")
    # Load dataset data
    with open(input_path, "r") as f:
        dataset_data = json.load(f)

    def answer_question_with_sliding_window(
        question, context, max_length=max_length, stride=128
    ):
        """
        Use sliding window to answer given question based on given context.

        :return: Answer to given question or an empty string if no answer
                 can be generated.
        """
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
            return_token_type_ids=return_token_type_ids,  # Dynamically set based on model config
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        if return_token_type_ids:
            token_type_ids = inputs["token_type_ids"].to(device)

        all_answers = []
        with torch.no_grad():
            for i in range(input_ids.size(0)):
                if return_token_type_ids:
                    outputs = model(
                        input_ids=input_ids[i].unsqueeze(0),
                        attention_mask=attention_mask[i].unsqueeze(0),
                        token_type_ids=token_type_ids[i].unsqueeze(0),
                    )
                else:
                    outputs = model(
                        input_ids=input_ids[i].unsqueeze(0),
                        attention_mask=attention_mask[i].unsqueeze(0),
                    )
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                start_scores = start_logits[0]
                end_scores = end_logits[0]
                answer_start = torch.argmax(start_scores)
                answer_end = torch.argmax(end_scores) + 1

                if answer_start < answer_end:  # Ensure the start is before the end
                    answer = tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(
                            input_ids[i][answer_start:answer_end]
                        )
                    )
                    all_answers.append(answer)

        best_answer = max(
            all_answers, key=len, default=""
        )  # Default to empty if no answers
        return best_answer

    logger.info("Generating answers for all articles in the dev set...")
    results = []

    # Process all articles
    for article in tqdm(dataset_data["data"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                answer = answer_question_with_sliding_window(question, context)
                results.append({"id": id_, "answer": answer})
        with open(output_path, "w") as f:
            json.dump({item["id"]: item["answer"] for item in results}, f)

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
