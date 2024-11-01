import numpy as np
from transformers import AutoTokenizer, TrainingArguments, Trainer, BertForSequenceClassification, EarlyStoppingCallback
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
dataset = load_dataset(
    "csv",
    data_files={
        "train": "../Datasets/TrainSentiment.csv",
        "validation": "../Datasets/ValidationSentiment.csv",
        "test": "../Datasets/TestSentiment.csv"  # Added missing comma
    },
)

# Checkpoint for BERT model
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenization function
def tokenize_fn(batch):
    return tokenizer(batch['sentence'], truncation=True)

# Tokenizing the dataset
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# Function to compute metrics
def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1-score": f1}

# Load model
model = BertForSequenceClassification.from_pretrained(
    checkpoint, num_labels=3
)

# Set training arguments
training_args = TrainingArguments(
    output_dir='../ClassifierModels/BERT',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Early stopping callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=1)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

# Train the model
trainer.train()
