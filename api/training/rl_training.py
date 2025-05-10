import json
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from datasets import Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

MODEL_PATH = "./models/fine-tuned-bart"
FEEDBACK_FILE = "./data/feedback_data.json"

# Load tokenizer
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)

# Convert model for PPO training (value head required)
ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(MODEL_PATH)

# ✅ Manually Set Generation Config (Fix for missing attribute)
ppo_model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)

# Create reference model (Frozen)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(MODEL_PATH)
ref_model.eval()  # Ensures it's not updated during training

# Load feedback data
def load_feedback():
    with open(FEEDBACK_FILE, "r") as f:
        return json.load(f)

# Reward function (higher rating = higher reward)
def compute_reward(rating):
    return (rating - 3) / 2  # Normalize between -1 to +1

# Prepare dataset
def prepare_dataset(feedback_data):
    return Dataset.from_list([
        {"query": item["question"], "reward": compute_reward(item["rating"])}
        for item in feedback_data
    ])

# Train using PPO
def train_with_rlhf():
    feedback_data = load_feedback()
    train_dataset = prepare_dataset(feedback_data)  # PPO requires a dataset

    # ✅ Corrected PPOConfig
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        mini_batch_size=4,
        gradient_accumulation_steps=4
    )

    # ✅ Define Reward Model (For now, using PPO model itself)
    reward_model = ppo_model  # Ideally, replace this with a trained reward model

    # ✅ Define Optimizers
    optimizer = Adam(ppo_model.parameters(), lr=ppo_config.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

    # ✅ Fixed PPOTrainer Initialization (Added `processing_class=tokenizer`)
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,  # ✅ Fix: Required parameter
        model=ppo_model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        optimizers=(optimizer, scheduler)
    )

    for item in train_dataset:
        inputs = tokenizer(item["query"], return_tensors="pt", truncation=True)
        query_tensor = inputs["input_ids"]

        # ✅ Fix: Generate response with manually set `generation_config`
        response_tensor = ppo_model.generate(query_tensor, max_length=50, pad_token_id=tokenizer.eos_token_id)
        reward = torch.tensor([item["reward"]])

        # Train with PPO step
        ppo_trainer.step(query_tensor, response_tensor, reward)

    # Save RLHF fine-tuned model
    ppo_model.save_pretrained("./models/rlhf_bart")

if __name__ == "__main__":
    train_with_rlhf()
