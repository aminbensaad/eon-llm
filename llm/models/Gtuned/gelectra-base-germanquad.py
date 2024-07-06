import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "scripts"))
sys.path.append(project_root)

import utils.predict as predict


def main(model_name, input_path, output_path):
    predict.generate_answers_with_sliding_window(
        model_name, input_path, output_path, use_token_type_ids=True, max_length=384
    )


if __name__ == "__main__":
    model_name = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    main(model_name, input_path, output_path)
