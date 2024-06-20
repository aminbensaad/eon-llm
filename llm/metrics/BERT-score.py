"""
BERTScore Evaluation Script

This script computes the BERTScore for model predictions against a reference dataset. 
BERTScore leverages BERT embeddings to evaluate the similarity of the generated text 
to the reference text, providing a more semantically informed metric than traditional 
word-overlap metrics.

Usage:
    python BERT-score.py <predictions_path> <dataset_path> <output_path> [-G]
    
    predictions_path: Path to the JSON file containing the model predictions.
    dataset_path: Path to the JSON file containing the reference dataset (SQuAD format).
    output_path: Path to the JSON file where the BERTScore results will be saved.
    -G: Optional flag to specify German language support.

Dependencies:
    bert-score (for BERTScore computation)
"""

import os
import json
import sys
import subprocess

# Ensure required packages are installed
# subprocess.check_call([sys.executable, "-m", "pip", "install", "bert-score"])

from bert_score import score


def load_data(predictions_file, references_file):
    with open(predictions_file, "r") as f:
        predictions = json.load(f)
    with open(references_file, "r") as f:
        references_data = json.load(f)

    references = []
    candidates = []

    for article in references_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                if isinstance(predictions, dict) and qid in predictions:
                    candidate_answer = predictions[qid]
                    references.append(qa["answers"][0]["text"] if qa["answers"] else "")
                    candidates.append(candidate_answer)
                elif isinstance(predictions, list) and qid in predictions:
                    references.append(qa["answers"][0]["text"] if qa["answers"] else "")
                    candidates.append(
                        qa["answers"][0]["text"]
                    )  # Use the reference as the prediction

    return candidates, references


def calculate_bertscore(candidates, references, lang):
    P, R, F1 = score(candidates, references, lang=lang, verbose=True)
    return {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item(),
    }


def main(predictions_path, dataset_path, output_path, lang):
    candidates, references = load_data(predictions_path, dataset_path)
    bertscore_results = calculate_bertscore(candidates, references, lang)

    with open(output_path, "w") as f:
        json.dump(bertscore_results, f, indent=4)

    print(f"BERTScore results saved to {output_path}")


if __name__ == "__main__":
    # Check for the optional -G flag
    is_german = "-G" in sys.argv
    if is_german:
        sys.argv.remove("-G")

    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 4:
        print(
            "Usage: python BERT-score.py <predictions_path> <dataset_path> <output_path> [-G]"
        )
        sys.exit(1)

    # Parse command line arguments
    predictions_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]

    # Set language for BERTScore
    lang = "de" if is_german else "en"

    # Run the main function
    main(predictions_path, dataset_path, output_path, lang)
