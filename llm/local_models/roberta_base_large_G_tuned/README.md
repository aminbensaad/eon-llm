# roberta-base-large-G-tuned

This is a fine-tuned model based on [roberta-large-squad2](https://huggingface.co/deepset/roberta-large-squad2).
It is fine-tuned on the SQuAD 1.1 dataset with the following hyperparameters:

```python
TrainingArguments(
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)
```
