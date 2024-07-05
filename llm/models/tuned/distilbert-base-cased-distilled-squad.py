import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the script can access utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from llm.scripts.utils import predict


def main(model_name, input_path, output_path):
    predict.generate_answers_with_sliding_window(
        model_name, input_path, output_path, use_token_type_ids=False
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
