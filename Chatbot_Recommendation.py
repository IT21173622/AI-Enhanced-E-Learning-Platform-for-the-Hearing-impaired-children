import random
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


# ------------------- API 6: Chatbot and Course Recommendation -------------------

# Load the CSV file
df = pd.read_csv('course.csv')

# Preprocess the data
df['questions'] = df['questions'].str.lower()
df['answers'] = df['answers'].str.lower()

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['questions'])

# Save the vectorizer and the model
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(df, 'data.pkl')

# Temporary storage for user questions and responses
user_data = {
    "questions": [],
    "answers": [],
    "levels": [],
    "subjects": []
}

# Pydantic model for request input
class UserQuestion(BaseModel):
    question: str

# Expanded feedback for Mathematics
math_feedback = {
    'Beginner': [
        "As a beginner, focus on mastering basic arithmetic operations like addition, subtraction, multiplication, and division. These are the building blocks for all future math topics.",
        "Start by understanding the concept of variables and simple algebraic equations. Practice solving equations step by step to build a strong foundation in algebra.",
        "Learn the basics of geometry, including shapes, angles, and area calculations. These concepts are essential for solving real-world problems and more advanced math topics."
    ],
    'Intermediate': [
        "At the intermediate level, work on solving more complex algebraic equations and inequalities. Practice factoring, expanding, and simplifying expressions to improve your problem-solving skills.",
        "Dive deeper into trigonometry by learning about sine, cosine, and tangent. These concepts are crucial for understanding waves, oscillations, and other real-world phenomena.",
        "Explore the basics of statistics and probability. Learn how to analyze data, calculate averages, and understand the likelihood of events occurring in different scenarios."
    ],
    'Advanced': [
        "Master calculus concepts like differentiation and integration. These tools are essential for understanding rates of change, areas under curves, and many real-world applications in science and engineering.",
        "Study linear algebra, including vectors, matrices, and linear transformations. These concepts are fundamental for computer graphics, machine learning, and solving systems of equations.",
        "Explore advanced topics like differential equations and stochastic processes. These are used to model complex systems in physics, biology, and finance."
    ],
    'General': [
        "Review foundational math concepts like arithmetic, algebra, and geometry. A strong understanding of these basics is essential for tackling more advanced topics.",
        "Practice problem-solving techniques for a variety of math topics. This will help you develop critical thinking skills and improve your ability to tackle unfamiliar problems.",
        "Explore both basic and advanced math topics to broaden your knowledge. This will help you see the connections between different areas of mathematics and their real-world applications."
    ]
}

# Expanded feedback for English
english_feedback = {
    'Beginner': [
        "As a beginner, focus on building a strong foundation in grammar. Learn about parts of speech, sentence structure, and common grammatical rules to improve your writing and speaking.",
        "Practice reading comprehension with simple texts. Try to summarize what you read and identify the main ideas, supporting details, and the author's purpose.",
        "Start building your vocabulary by learning common words and their meanings. Use flashcards or apps to practice and review new words regularly."
    ],
    'Intermediate': [
        "At the intermediate level, work on writing clear and structured essays. Practice organizing your ideas into paragraphs and using transitions to connect your thoughts logically.",
        "Improve your reading comprehension by analyzing texts for tone, mood, and themes. Try to understand the author's perspective and how they convey their message.",
        "Expand your vocabulary by learning synonyms, antonyms, and word roots. This will help you understand and use more complex language in your writing and speaking."
    ],
    'Advanced': [
        "Master advanced writing techniques like persuasive and creative writing. Practice crafting compelling arguments and using literary devices to enhance your storytelling.",
        "Focus on critical analysis of literature and poetry. Learn to interpret symbolism, themes, and the author's use of language to convey deeper meanings.",
        "Study advanced vocabulary and etymology to improve your writing. Understanding the origins of words can help you use them more effectively and expand your linguistic knowledge."
    ],
    'General': [
        "Review foundational English skills like grammar, vocabulary, and sentence structure. A strong grasp of these basics is essential for effective communication.",
        "Practice both reading and writing to improve your overall proficiency. Try to read a variety of texts and write regularly to develop your skills.",
        "Explore different genres of literature to broaden your understanding of English. This will help you appreciate different writing styles and improve your own writing."
    ]
}

# Function to recommend courses based on level and subject
def recommend_courses(level, subject):
    math_courses = {
        'Beginner': ["Basic Algebra", "Introduction to Geometry", "Pre-Algebra"],
        'Intermediate': ["Advanced Algebra", "Trigonometry", "Intermediate Geometry"],
        'Advanced': ["Calculus", "Linear Algebra", "Advanced Geometry"],
        'General': ["General Mathematics", "Introduction to Mathematics"]
    }
    english_courses = {
        'Beginner': ["English Grammar Basics", "Basic Reading Comprehension"],
        'Intermediate': ["Essay Writing", "Intermediate Reading Comprehension"],
        'Advanced': ["Creative Writing", "Advanced Reading and Critical Analysis"],
        'General': ["General English", "Introduction to English Literature"]
    }
    if subject == "Mathematics":
        return math_courses.get(level, math_courses['General'])
    elif subject == "English":
        return english_courses.get(level, english_courses['General'])
    else:
        return math_courses.get(level, math_courses['General']) + english_courses.get(level, english_courses['General'])

# Function to provide feedback based on level and subject
def provide_feedback(level, subject):
    if subject == "Mathematics":
        feedback_options = math_feedback.get(level, math_feedback['General'])
    elif subject == "English":
        feedback_options = english_feedback.get(level, english_feedback['General'])
    else:
        feedback_options = math_feedback.get(level, math_feedback['General']) + english_feedback.get(level, english_feedback['General'])
    
    # Randomly select one feedback from the options
    return random.choice(feedback_options)

# Function to interact with the chatbot
def chatbot(question):
    vectorizer = joblib.load('vectorizer.pkl')
    df = joblib.load('data.pkl')
    question = question.lower()
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, X)
    most_similar_idx = similarities.argmax()
    answer = df.iloc[most_similar_idx]['answers']
    level = df.iloc[most_similar_idx]['level']
    subject = df.iloc[most_similar_idx]['subject']
    return answer, level, subject

# Endpoint to ask questions one by one
@app.post("/chatbot")
def ask_question(user_question: UserQuestion):

    answer, level, subject = chatbot(user_question.question)
    user_data["questions"].append(user_question.question)
    user_data["answers"].append(answer)
    user_data["levels"].append(level)
    user_data["subjects"].append(subject)

    return {
        "question": user_question.question,
        "answer": answer,
        "level": level,
        "subject": subject,
        "questions_asked": len(user_data["questions"])
    }

# Endpoint to get recommendations and feedback
@app.get("/recommendations")
def get_recommendations():
    if len(user_data["questions"]) < 5:
        raise HTTPException(status_code=400, detail="Please ask 5 questions.")

    user_level = max(set(user_data["levels"]), key=user_data["levels"].count)
    user_subject = max(set(user_data["subjects"]), key=user_data["subjects"].count)

    recommendations = recommend_courses(user_level, user_subject)
    feedback = provide_feedback(user_level, user_subject)

    # Clear user data after providing recommendations
    user_data["questions"].clear()
    user_data["answers"].clear()
    user_data["levels"].clear()
    user_data["subjects"].clear()

    return {
        "level": user_level,
        "subject": user_subject,
        "recommended_courses": recommendations,
        "feedback": feedback
    }


# ------------------- API 7: Quiz Score to Course Recommendation -------------------
# Load the saved model and data
vectorizer = joblib.load('tfidf_vectorizer.pkl')
courses_df = joblib.load('course_recommendation_dataset.pkl')

# Ensure the 'tags_string' column exists in the DataFrame
if 'tags_string' not in courses_df.columns:
    courses_df['tags_string'] = courses_df['course_name']  # Use 'course_name' as a proxy for tags

# Define a Pydantic model for the input data
class QuizScore(BaseModel):
    english_score: float
    math_score: float

# Helper function to map score to level
def score_to_level(score):
    level_mapping = {
        range(0, 40): 'Beginner',
        range(40, 70): 'Intermediate',
        range(70, 101): 'Advanced'
    }
    return next(level for score_range, level in level_mapping.items() if score in score_range)

# Helper function to recommend courses based on score
def recommend_courses_by_level(subject, score):
    # Filter courses by subject
    subject_courses = courses_df[courses_df['subject'] == subject]
    
    # Map the quiz score to a course level
    level = score_to_level(score)
    
    # Filter courses by the level determined
    level_courses = subject_courses[subject_courses['level'] == level]
    
    # Vectorize the filtered level courses for cosine similarity calculation
    level_vectors = vectorizer.transform(level_courses['tags_string'])
    
    # Normalize the student's score to match with the vector representation
    normalized_score = np.array([score / 100.0])
    score_vector = np.zeros(level_vectors.shape[1])
    score_vector[:len(normalized_score)] = normalized_score

    # Calculate cosine similarity between the student's score and each course's vector
    similarity_scores = cosine_similarity([score_vector], level_vectors)

    # Get the indices of the most similar courses
    similar_courses_indices = similarity_scores.argsort()[0][::-1]  # Sort in descending order

    # Return the top 3 recommended courses
    recommended_courses = level_courses.iloc[similar_courses_indices[:3]]
    return recommended_courses[['course_name', 'subject', 'level']]

# FastAPI endpoint to predict courses
@app.post("/quiz_recommend_courses/")
def recommend_courses_quiz(quiz_scores: QuizScore):
    # Recommend English courses
    recommended_english_courses = recommend_courses_by_level('English', quiz_scores.english_score)
    
    # Recommend Math courses
    recommended_math_courses = recommend_courses_by_level('Mathematics', quiz_scores.math_score)

    return {
        "recommended_english_courses": recommended_english_courses.to_dict(orient="records"),
        "recommended_math_courses": recommended_math_courses.to_dict(orient="records")
    }
