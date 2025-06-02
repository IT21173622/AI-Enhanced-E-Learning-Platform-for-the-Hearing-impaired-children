
# Constants
NUM_SAMPLES = 100000  # Adjust the number of data points you want to generate
MAX_ROUNDS = 10  # Maximum rounds per level
MAX_SCORE_RANGE = 100  # Maximum score (XP) range
MAX_TIME_FOR_MAX_SCORE = 60  # Maximum time (second) for achieving max score
game_level=15
max_engagement_time=180 #second


# Function to calculate game score (XP) based on your logic
def calculate_game_score(success_count, attempt_count, engagement_time_total, max_rounds):
    # Calculate the maximum score based on your criteria
    max_score = MAX_SCORE_RANGE

    # Calculate the time threshold for achieving max score
    #max_time_threshold = max(1, max_rounds / MAX_TIME_FOR_MAX_SCORE)
    max_time_threshold_per_attend =engagement_time_total / attempt_count 

    # Calculate score based on your criteria
    if success_count == attempt_count and max_time_threshold_per_attend  <= MAX_TIME_FOR_MAX_SCORE:
        score = max_score
    elif success_count <= attempt_count and max_time_threshold_per_attend  <= MAX_TIME_FOR_MAX_SCORE:
        score = ((success_count / attempt_count) * max_score*0.5)+(max_score*0.5)

    elif  success_count <= attempt_count and max_time_threshold_per_attend >= MAX_TIME_FOR_MAX_SCORE:
        score = ((MAX_TIME_FOR_MAX_SCORE / max_time_threshold_per_attend) * max_score*0.5)+((success_count / attempt_count) * max_score*0.5)

    score=min(score,max_score)

    return int(score)
