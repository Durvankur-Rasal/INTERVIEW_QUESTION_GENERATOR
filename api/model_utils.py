from transformers import BartForConditionalGeneration, BartTokenizer

# Load Fine-tuned Model
MODEL_PATH = "./models/fine-tuned-bart"  # Adjust path if needed
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)

def generate_questions(job_title, job_description, num_questions=5):
    input_text = f"Generate diverse interview questions for {job_title}. Description: {job_description}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                       padding="max_length", max_length=512)

    # Use sampling mode to generate diverse outputs
    output = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=num_questions,
        do_sample=True,  # Enables randomness
        temperature=1.2,  # Controls diversity
        top_k=50,  # Limits token choices
        top_p=0.95,  # Nucleus sampling
        repetition_penalty=1.5,  # Avoids repetitive questions
        early_stopping=True,  # Stops generation early if needed
        num_beams=1,  # Ensures pure sampling without beam search conflict
    )

    questions = [tokenizer.decode(q, skip_special_tokens=True) for q in output]
    return list(set(questions))  # Ensures unique questions
