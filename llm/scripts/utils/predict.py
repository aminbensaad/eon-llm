import logging
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name):
    logger.info("Loading the tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def load_pipeline(model_name):
    logger.info("Loading the question-answering pipeline...")
    qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return qa_pipeline


def load_dataset(input_path):
    logger.info(f"Loading dataset from {input_path}...")
    with open(input_path, "r", encoding='utf-8') as f:
        dataset_data = json.load(f)
    return dataset_data


def save_results(output_path, results):
    logger.info(f"Saving the generated answers to {output_path}...")
    with open(output_path, "w") as f:
        json.dump({item["id"]: item["answer"] for item in results}, f)
    logger.info("Answers saved successfully.")


def answer_question_with_sliding_window(
    tokenizer,
    model,
    device,
    question,
    context,
    use_token_type_ids=True,
    max_length=512,
    stride=128,
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
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    if use_token_type_ids:
        token_type_ids = inputs["token_type_ids"].to(device)
    else:
        token_type_ids = None

    all_answers = []
    with torch.no_grad():
        if token_type_ids is not None:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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


def generate_answers_with_sliding_window(
    model_name, input_path, output_path, use_token_type_ids=True, max_length=512
):
    tokenizer, model, device = load_model_and_tokenizer(model_name)
    dataset_data = load_dataset(input_path)
    results = []

    logger.info("Generating answers for all articles in the dataset...")
    for article in tqdm(dataset_data["data"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id_ = qa["id"]
                answer = answer_question_with_sliding_window(
                    tokenizer,
                    model,
                    device,
                    question,
                    context,
                    use_token_type_ids,
                    max_length,
                )
                results.append({"id": id_, "answer": answer})

    save_results(output_path, results)


def generate_answers_with_pipeline(model_name, input_path, output_path):
    qa_pipeline = load_pipeline(model_name)
    dataset_data = load_dataset(input_path)
    results = []

    logger.info("Generating answers for all questions in the dataset...")
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

    save_results(output_path, results)


def validate_predictions(predictions_path, dataset_path):
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)
        dataset = dataset_json["data"]

    # Initialize sets to collect IDs
    dataset_ids = set()
    pred_ids = set(predictions.keys())

    # Collect all IDs from the dataset
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = str(qa["id"])
                dataset_ids.add(qid)

    # Find missing IDs
    missing_from_preds = dataset_ids - pred_ids
    missing_from_dataset = pred_ids - dataset_ids

    print(f"Total IDs in dataset: {len(dataset_ids)}")
    print(f"Total IDs in predictions: {len(pred_ids)}")
    print(f"Missing IDs in predictions: {len(missing_from_preds)}")
    print(f"Missing IDs in dataset: {len(missing_from_dataset)}")

    if missing_from_preds:
        print(f"Missing from predictions: {missing_from_preds}")
    if missing_from_dataset:
        print(f"Missing from dataset: {missing_from_dataset}")

    # Check prediction format
    for qid, answer in predictions.items():
        if not isinstance(qid, str) or not isinstance(answer, str):
            print(f"Invalid prediction format for ID: {qid}, Answer: {answer}")
            return False

    return True
