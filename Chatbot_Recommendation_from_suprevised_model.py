from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Initialize FastAPI app
app = FastAPI()

# In-memory store for user sessions (no user_id, just an internal session)
user_sessions = {}

# Load the pre-trained model
def load_model():
    """Load the pre-trained model."""
    return joblib.load('chatbot_model.pkl')

# Load the courses CSV
def load_courses(csv_path):
    """Load courses from a CSV file."""
    return pd.read_csv(csv_path)

# Prepare training data
def prepare_training_data(csv_path):
    """Load training data from the provided CSV."""
    return pd.read_csv(csv_path)

# Detect subject and level using the pre-trained model
def detect_level_and_subject(user_questions, model):
    """Use the pre-trained model to classify the subject and level."""
    predictions = model.predict(user_questions)
    subjects_levels = [pred.split('|') for pred in predictions]
    subjects = [sl[0] for sl in subjects_levels]
    levels = [sl[1] for sl in subjects_levels]

    # Majority voting for subject and level
    subject = max(set(subjects), key=subjects.count)
    level = max(set(levels), key=levels.count)

    return subject, level

# Answer user questions based on training data
def answer_user_questions(user_questions, training_data):
    """Provide answers to user questions based on training data."""
    answers = []
    for question in user_questions:
        matched = training_data[training_data['questions'].str.contains(question, case=False, na=False)]
        if not matched.empty:
            answers.append(matched.iloc[0]['answers'])
        else:
            answers.append("Sorry, I don't have an answer for that question.")
    return answers

# Recommend courses based on subject and level
def recommend_courses(subject, level, courses_df):
    """Recommend courses based on the detected subject and level."""
    filtered_courses = courses_df[(courses_df['Area'] == subject) & (courses_df['Difficulty Level'] == level)]

    if filtered_courses.empty:
        return "No courses found for your selected subject and level."

    recommendations = []
    for _, course in filtered_courses.iterrows():
        recommendations.append(f"Course: {course['Course Title']}\nDescription: {course.get('Content Keywords', 'No description available.')}")

    return "\n".join(recommendations)

# Pydantic model to handle incoming request body
class UserQuestion(BaseModel):
    question: str

# Main API Endpoint to interact with the user
@app.post("/ask_question/")
async def ask_question(user_question: UserQuestion):
    """
    Handle user questions one at a time.
    User sends one question, and the API responds with an answer.
    """
    # Initialize a session for a user if it doesn't exist yet
    if "questions" not in user_sessions:
        user_sessions["questions"] = []
        user_sessions["answers"] = []

    # Add the user's question to the session
    user_sessions["questions"].append(user_question.question)

    # Load training data
    training_data_path = "2000_questions_dataset.csv"
    training_data = prepare_training_data(training_data_path)

    # Provide an immediate answer to the question
    answer = answer_user_questions([user_question.question], training_data)[0]
    user_sessions["answers"].append(answer)

    # Respond immediately after each question
    return {"answer": answer, "message": "Your question has been answered."}

# Final endpoint to detect subject, level, and recommend courses
@app.post("/get_course_recommendations/")
async def get_course_recommendations():
    """
    After five questions, detect the subject and level, and recommend courses.
    """
    # Check if the user has asked 5 questions
    if len(user_sessions.get("questions", [])) < 5:
        raise HTTPException(status_code=400, detail="You must ask exactly 5 questions first.")

    # Get the user's questions
    user_questions = user_sessions["questions"]

    # Load the pre-trained model
    model = load_model()

    # Load courses and training data
    courses_csv_path = "course_recommendation_dataset.csv"
    courses_df = load_courses(courses_csv_path)

    # Detect the subject and level using the pre-trained model
    subject, level = detect_level_and_subject(user_questions, model)

    # Recommend courses based on the detected subject and level
    recommendations = recommend_courses(subject, level, courses_df)

    # Clear the session after processing
    user_sessions.clear()

    return {
        "subject": subject,
        "level": level,
        "course_recommendations": recommendations
    }

# Run the app using `uvicorn`:
# uvicorn main:app --reload
