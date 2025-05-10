from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Interview Question Generator API!"}

def test_generate_questions():
    sample_input = {
        "job_title": "Software Engineer",
        "job_description": "Experience in Python, Machine Learning, and APIs.",
        "num_questions": 3
    }
    response = client.post("/generate/", json=sample_input)
    assert response.status_code == 200
    assert "questions" in response.json()
    assert isinstance(response.json()["questions"], list)
    assert len(response.json()["questions"]) == 3

def test_generate_questions_missing_input():
    response = client.post("/generate/", json={})
    assert response.status_code == 422  # Expecting 422 instead of 400

def test_generate_questions_invalid_input():
    response = client.post("/generate/", json={"job_title": 123, "job_description": True, "num_questions": "five"})
    assert response.status_code == 422  # FastAPI will return a validation error
