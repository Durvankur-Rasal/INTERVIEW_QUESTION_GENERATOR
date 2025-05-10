from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_utils import generate_questions
from feedback import router as feedback_router
import os
import logging

app = FastAPI()

# Allow frontend access (supports both local & deployed environments)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # ✅ Supports multiple origins from env variable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for question generation
class QuestionRequest(BaseModel):
    job_title: str
    job_description: str
    num_questions: int = 5

@app.post("/generate/")
def generate(request: QuestionRequest):
    try:
        questions = generate_questions(request.job_title, request.job_description, request.num_questions)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

# ✅ Include feedback router with prefix
logging.debug("Mounting feedback router...")
app.include_router(feedback_router)
logging.debug("Feedback router mounted successfully")

@app.get("/")
def home():
    return {"message": "Welcome to the Interview Question Generator API!"}
