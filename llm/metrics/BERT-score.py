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

import json
import sys
from bert_score import score


def load_data(predictions_file, references_file):
    with open(predictions_file, "r") as f:
        predictions = json.load(f)
    with open(references_file, "r") as f:
        references_data = json.load(f)

    references = []
    candidates = []
    has_ans_references = []
    has_ans_candidates = []
    no_ans_references = []
    no_ans_candidates = []

    for article in references_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = str(qa["id"])
                if qid in predictions:
                    candidate_answer = predictions[qid]
                    reference_answer = qa["answers"][0]["text"] if qa["answers"] else ""
                    references.append(reference_answer)
                    candidates.append(candidate_answer)

                    if qa["is_impossible"]:
                        no_ans_references.append(reference_answer)
                        no_ans_candidates.append(candidate_answer)
                    else:
                        has_ans_references.append(reference_answer)
                        has_ans_candidates.append(candidate_answer)
                else:
                    references.append("")
                    candidates.append("")
                    if qa["is_impossible"]:
                        no_ans_references.append("")
                        no_ans_candidates.append("")
                    else:
                        has_ans_references.append("")
                        has_ans_candidates.append("")

    return (
        candidates,
        references,
        has_ans_candidates,
        has_ans_references,
        no_ans_candidates,
        no_ans_references,
    )


def replace_special_tokens(
    texts, special_tokens=["[CLS]", "[SEP]", "[PAD]"], replacement=""
):
    return [replacement if text in special_tokens else text for text in texts]


def calculate_bertscore(candidates, references, lang):
    # Replace special tokens with an empty string or any placeholder text
    candidates = replace_special_tokens(candidates)
    references = replace_special_tokens(references)

    if not candidates or not references:
        print("Error: Candidates or references list is empty.")
        return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    # Debug: print the first few candidates and references
    print("First few candidates:", candidates[:5])
    print("First few references:", references[:5])

    P, R, F1 = score(candidates, references, lang=lang, verbose=True)
    return {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item(),
    }


def main(predictions_path, dataset_path, output_path, lang):
    (
        candidates,
        references,
        has_ans_candidates,
        has_ans_references,
        no_ans_candidates,
        no_ans_references,
    ) = load_data(predictions_path, dataset_path)

    bertscore_results = calculate_bertscore(candidates, references, lang)
    has_ans_bertscore_results = calculate_bertscore(
        has_ans_candidates, has_ans_references, lang
    )
    no_ans_bertscore_results = calculate_bertscore(
        no_ans_candidates, no_ans_references, lang
    )

    results = {
        "BERTScore": bertscore_results,
        "HasAns_BERTScore": has_ans_bertscore_results,
        "NoAns_BERTScore": no_ans_bertscore_results,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"BERTScore results saved to {output_path}")


if __name__ == "__main__":
    # Check for the optional -G flag to process GermanQuAD
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

    main(predictions_path, dataset_path, output_path, lang)
