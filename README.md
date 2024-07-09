# Knowledge querying with open-source LLMs

This is the code repository of group 9 for the E.ON data challenge of the
course "Data Analytics in Applications" of Technical University of Munich.

## Setup

To get started, clone the repository first:

```sh
git clone https://github.com/aminbensaad/eon-llm.git
```

In order to access our own fine-tuned models, instructions to install them
can be found in `llm/local_models/*/README.md`.

To avoid conflicts with the system, it is recommended to use a virtual environment,
for example with `conda` or `virtualenv`.
As an example for `conda`, the following two commands can be used:

```sh
conda env create -f environment.yml
conda activate eon-llm
```

Finally, install all the required dependencies with the following command:

```sh
pip install -r requirements.txt
```

## Structure

The repository is divided into an evaluation pipeline, a chatbot UI,
a fine-tuning script and Jupyter notebooks with experiments.

## Evaluation Pipeline

Before a model can be evaluated, the predictions have to be generated first.
This can be done with the following command for the SQuAD dataset on
fine-tuned models:

```sh
python llm/scripts/run.py -p -d SQuAD -m tuned
```

To use GermanQuAD instead of SQuAD replace "SQuAD" in the command above by "G"
and for different model sets the following categories can be used instead of "tuned":

- tuned: SQuAD fine-tuned models
- base: Untuned models
- Gtuned: GermanQuAD fine-tuned models
- Gbase: Untuned models for German

To run the evaluation scripts on the generated results, run the following command:

```sh
python llm/scripts/run.py -e --all -d SQuAD -m tuned
```

The same modifications as with the command before this one can be used to run
the evaluations on different models or datasets.

The results can be found in the directory of the evaluation pipeline in `model_results/`.

## Chatbot UI

Located in `chatbot-ui/`, a graphical interface can be found to interact with the
evaluated models.
It can be run with the following command:

```sh
streamlit run chatbot-ui/Chatbot.py
```

## Fine-tuning script

The fine-tuning script is located at `./fine-tune.py` and can be run with the
following command:

```sh
python fine-tune.py
```

All adjustments to selected model, used dataset and hyperparameters must be made
in the script itself.

## Experiments

The Jupyter notebooks contain various code snippets to generate figures, run
inference or evaluate results.
The following notebooks exist:

- `llm/experiments.ipynb`: Code to generate the figures used to compare models
                           which were also used in presentation and paper
