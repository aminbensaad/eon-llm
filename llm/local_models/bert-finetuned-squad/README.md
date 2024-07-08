# bert-finetuned-squad

This is a fine-tuned model based on [bert-base-cased](https://huggingface.co/google-bert/bert-base-cased).
It is fine-tuned on the SQuAD 1.1 dataset with the following hyperparameters:

```python
TrainingArguments(
    evaluation_strategy="no",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
)
```
