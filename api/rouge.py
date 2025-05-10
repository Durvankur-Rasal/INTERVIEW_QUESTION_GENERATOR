import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import json

# Load Fine-Tuned BART Model & Tokenizer
MODEL_PATH = "./models/fine-tuned-bart"  # Adjust if needed
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)

# Load Test Data (Reference Questions)
TEST_DATA_PATH = "./data/test_data.json"  # JSON file containing job titles, descriptions, and reference questions
with open(TEST_DATA_PATH, "r") as f:
    test_data = json.load(f)

# Initialize ROUGE Scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Evaluate ROUGE Score
rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

for item in test_data:
    job_title = item["job_title"]
    job_description = item["job_description"]
    reference_question = item["reference_question"]  # Ground truth question

    # Generate Interview Question
    input_text = f"Generate interview questions for {job_title}. Description: {job_description}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=100, num_beams=5)
    
    generated_question = tokenizer.decode(output[0], skip_special_tokens=True)

    # Compute ROUGE Scores
    scores = scorer.score(reference_question, generated_question)
    rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
    rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
    rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

# Calculate Average ROUGE Scores
avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

# Print Results
print(f"Average ROUGE-1: {avg_rouge1:.4f}")
print(f"Average ROUGE-2: {avg_rouge2:.4f}")
print(f"Average ROUGE-L: {avg_rougeL:.4f}")
