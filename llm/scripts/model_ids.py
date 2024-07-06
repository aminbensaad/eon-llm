import os

base_path = ".."
local_model_dir = os.path.join(base_path, "local_models")
model_dir = os.path.join(base_path, "models")

model_IDs = {
    "base": [  # German base models (pre-trained)
        "TheBloke/mistral-ft-optimized-1227-GGUF",
        "tiiuae/falcon-7b-instruct",  # ✅
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "nvidia/Llama3-ChatQA-1.5-8B",  # ✅
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # PENDING access
        "Deci/DeciLM-7B-instruct",
        # HuggingFace Leaderboard
        "BarraHome/Mistroll-7B-v2.2",
        "yam-peleg/Experiment26-7B",
        "MTSAIR/multi_verse_model",
    ],
    "Gbase": [  # German base models (pre-trained)
        "philschmid/instruct-igel-001",
        "TheBloke/em_german_leo_mistral-GGUF",
        "VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct",
        "TheBloke/DiscoLM_German_7b_v1-GGUF",
    ],
    "tuned": [  # General models (fine-tuned on SQuAD)
        "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",  # ✅
        "distilbert/distilbert-base-cased-distilled-squad",  # ✅
        "timpal0l/mdeberta-v3-base-squad2",  # ✅
        "deepset/roberta-base-squad2",  # ✅
        "deepset/roberta-large-squad2",  # ✅
        "deepset/xlm-roberta-base-squad2",
        "deepset/tinyroberta-squad2",
    ],
    "Gtuned": [  # German models (fine-tuned on GermanQuAD)
        "deutsche-telekom/bert-multi-english-german-squad2",  # ✅
        "deepset/gelectra-base-germanquad-distilled",
        "deepset/gelectra-base-germanquad",  # ✅
        "deepset/gelectra-large-germanquad",  # ✅
    ],
    "local": [
        "bert-finetuned-squad/checkpoint-33276",  # ✅
    ],
}


def model_name_from_id(model_id: str, model_type: str = "") -> str:
    model_id_components = model_id.split("/")
    if model_type == "local":
        return model_id_components[-2]
    return model_id_components[-1]


def model_script_path(model_type: str, model_id: str) -> str:
    model_name = model_name_from_id(model_id)
    script_path = os.path.join(model_dir, model_type, f"{model_name}.py")

    if os.path.exists(script_path):
        return script_path

    if "base" in model_type:
        fallback_type = "base"
    elif "tuned" in model_type or "local" in model_type:
        fallback_type = "tuned"
    else:
        fallback_type = model_type

    return os.path.join(base_path, "scripts", f"{fallback_type}.py")
