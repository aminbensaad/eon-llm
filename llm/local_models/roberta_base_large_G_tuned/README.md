# roberta-base-large-G-tuned

This is a fine-tuned model based on [roberta-large-squad2](https://huggingface.co/deepset/roberta-large-squad2).

## Installation

Download the model files from the following link:

[Google Drive 'checkpoint'](https://drive.google.com/drive/folders/1Tw3MaHUJVkHi66GHMO_a1M2GSwZ8HF8y?usp=sharing)

And place all the contained files in 'checkpoint/' in this directory.

## Hyperparameters

It is fine-tuned on the SQuAD 1.1 dataset with the following hyperparameters:

```python
TrainingArguments(
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)
```
