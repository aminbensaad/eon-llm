import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the script can access utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.utils import utils

if __name__ == "__main__":
    model_name = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    utils.generate_answers_with_sliding_window(
        model_name, input_path, output_path, use_token_type_ids=True
    )
