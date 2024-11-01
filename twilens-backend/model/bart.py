import numpy as np
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric

# Load the dataset
dataset = load_dataset(
    "csv",
    data_files={
        "train": "../Datasets/TrainSummary.csv",
        "validation": "../Datasets/ValidationSummary.csv",
        "test": "../Datasets/TestSummary.csv"
    },
)

# Initialize the tokenizer
checkpoint = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Set maximum input and target lengths
max_input = 1024
max_target = 128

# Tokenization function
def tokenize_fn(batch):
    model_inputs = tokenizer(batch['text'], max_length=max_input, padding='max_length', truncation=True)

    with tokenizer.as_target_tokenizer():
        targets = tokenizer(batch['summary'], max_length=max_target, padding='max_length', truncation=True)
        
    model_inputs['labels'] = targets['input_ids']
    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['summary'])

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Create a data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Load the metric
metric = load_metric("rouge")

# Compute metrics function
def compute_metrics(pred):
    pred_ids = pred.predictions
    labels_ids = pred.label_ids

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    res = metric.compute(
        predictions=pred_str,
        references=label_str,
        use_stemmer=True
    )
    res = {key: value.mid.fmeasure * 100 for key, value in res.items()}

    pred_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in pred_ids
    ]
    res["gen_len"] = np.mean(pred_lens)

    return {k: round(v, 4) for k, v in res.items()}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='../SummarizerModels/BART',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    eval_accumulation_steps=1,
    fp16=True
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
