from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os
import logging

# Define FastAPI router with a prefix
router = APIRouter(
    prefix="/feedback",
    tags=["feedback"]
)

# Feedback storage file
FEEDBACK_FILE = "./data/feedback_data.json"

# Feedback Model
class Feedback(BaseModel):
    question: str
    rating: int

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load existing feedback (if file exists)
def load_feedback():
    logging.debug(f"Attempting to load feedback from {FEEDBACK_FILE}")
    if not os.path.exists(FEEDBACK_FILE):  # ✅ Create file if missing
        logging.debug("Feedback file doesn't exist, creating new file")
        with open(FEEDBACK_FILE, "w") as f:
            json.dump([], f)
    try:
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)
            logging.debug(f"Successfully loaded {len(data)} feedback entries")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading feedback: {str(e)}")
        return []

# Save feedback to JSON file
def save_feedback(feedback_data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=4)

@router.post("/submit-feedback")
async def submit_feedback(feedback: Feedback):
    logging.debug(f"Received feedback submission: {feedback.dict()}")
    try:
        feedback_data = load_feedback()
        
        # Append new feedback
        feedback_data.append({"question": feedback.question, "rating": feedback.rating})
        logging.debug(f"Added new feedback. Total entries: {len(feedback_data)}")
        
        # Save updated feedback
        save_feedback(feedback_data)
        logging.debug("Successfully saved feedback to file")

        return {"message": "Feedback submitted successfully", "question": feedback.question, "rating": feedback.rating}
    except Exception as e:
        logging.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-feedback")
async def get_feedback():
    feedback_data = load_feedback()
    return {"feedback": feedback_data}
