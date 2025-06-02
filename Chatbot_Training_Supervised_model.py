import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the CSV file containing course data
def load_courses(csv_path):
    """
    Load courses from a CSV file.
    """
    return pd.read_csv(csv_path)

# Prepare training data for subject and level detection
def prepare_training_data(csv_path):
    """
    Load training data from the provided CSV.
    """
    return pd.read_csv(csv_path)

# Train and save the model
def train_and_save_model(training_data_path):
    """
    Train a model using the provided dataset and save it.
    """
    # Load the training data
    data = prepare_training_data(training_data_path)

    # Combine subject and level into a single target
    data['target'] = data['subject'] + '|' + data['level']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['questions'], data['target'], test_size=0.2, random_state=42
    )

    # Build a pipeline with TF-IDF and Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(pipeline, 'chatbot_model.pkl')
    print("Model trained and saved as 'chatbot_model.pkl'.")

# Load the pre-trained model
def load_model():
    """
    Load the pre-trained model.
    """
    return joblib.load('chatbot_model.pkl')

# Detect subject and level using the pre-trained model
def detect_level_and_subject(user_questions, model):
    """
    Use the pre-trained model to classify the subject and level.
    """
    predictions = model.predict(user_questions)
    subjects_levels = [pred.split('|') for pred in predictions]
    subjects = [sl[0] for sl in subjects_levels]
    levels = [sl[1] for sl in subjects_levels]

    # Majority voting for subject and level
    subject = max(set(subjects), key=subjects.count)
    level = max(set(levels), key=levels.count)

    return subject, level

# Answer user questions based on the training data
def answer_user_questions(user_questions, training_data):
    """
    Provide answers to user questions based on training data.
    """
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
    """
    Recommend courses based on the detected subject and level.
    """
    filtered_courses = courses_df[(courses_df['Area'] == subject) & (courses_df['Difficulty Level'] == level)]

    if filtered_courses.empty:
        return "No courses found for your selected subject and level."

    # Generate recommendations
    recommendations = []
    for _, course in filtered_courses.iterrows():
        recommendations.append(f"Course: {course['Course Title']}\nDescription: {course.get('Content Keywords', 'No description available.')}")

    return "\n".join(recommendations)

# Main Chatbot Function
def chatbot():
    # Paths for datasets
    training_data_path = "meaningful_2000_questions_dataset.csv"
    courses_csv_path = "course_recommendation_dataset.csv"

    # Load the courses CSV
    courses_df = load_courses(courses_csv_path)

    # Load the pre-trained model
    model = load_model()

    # Load training data
    training_data = prepare_training_data(training_data_path)

    user_questions = []
    print("Welcome to the chatbot! Please ask 5 questions related to Math or English.")
    print("I will analyze your questions, answer them, and recommend relevant courses based on your expertise.")

    # Interact with the user and collect 5 questions
    for i in range(5):
        user_question = input(f"Question {i + 1}: ")
        user_questions.append(user_question)

    # Provide answers to user questions
    answers = answer_user_questions(user_questions, training_data)

    print("\nHere are the answers to your questions:")
    for i, answer in enumerate(answers, start=1):
        print(f"Answer {i}: {answer}")

    # Detect subject and level using the pre-trained model
    subject, level = detect_level_and_subject(user_questions, model)

    # Recommend courses based on detected subject and level
    recommendations = recommend_courses(subject, level, courses_df)

    # Output course recommendations
    print(f"\nBased on your questions, I detected the following subject and level:")
    print(f"Subject: {subject}\nLevel: {level}")
    print("\nHere are some course recommendations for you: ")
    print(recommendations)

if __name__ == "__main__":
    # Train the model with the 2000-question dataset
    train_and_save_model("meaningful_2000_questions_dataset.csv")
    chatbot()
