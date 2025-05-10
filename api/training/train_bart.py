import torch
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Dataset (Replace with your actual dataset path)
df = pd.read_csv("merged_dataset.csv")  # Ensure columns exist in CSV

# Preprocessing - Creating Input-Output Pairs
df["input_text"] = "Generate interview questions for " + df["job_title"] + ". Description: " + df["job_description"]
df["output_text"] = df["interview_questions"]

# Split Data (Train: 80%, Validation: 20%)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["input_text"].tolist(), df["output_text"].tolist(), test_size=0.2, random_state=42
)

# Load Tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["input_text"], text_target=examples["output_text"], truncation=True, padding="max_length", max_length=512)

train_dataset = Dataset.from_dict({"input_text": train_texts, "output_text": train_labels}).map(tokenize_function, batched=True)
val_dataset = Dataset.from_dict({"input_text": val_texts, "output_text": val_labels}).map(tokenize_function, batched=True)

# Load Pre-trained BART Model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Training Arguments
training_args = TrainingArguments(
    output_dir="./bart-interview-gen",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Fine-tune Model
trainer.train()

# Save Model & Tokenizer
model.save_pretrained("fine-tuned-bart")
tokenizer.save_pretrained("fine-tuned-bart")

print("Fine-tuning complete! Model saved.")
