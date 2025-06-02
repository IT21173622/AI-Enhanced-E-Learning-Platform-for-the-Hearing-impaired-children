import pandas as pd
import numpy as np
import os
import joblib
import math
import warnings
import pmdarima as pm

from pmdarima.model_selection import train_test_split
from pmdarima.arima.utils import ndiffs
from datetime import datetime
from sklearn.preprocessing import StandardScaler
#from .firebase_data import fire_base_player_game


current_dir=os.path.dirname(os.path.abspath(__file__))

# Load the saved model and scalers

LR_model=joblib.load(os.path.join(current_dir,"model/improvement_score/LR_model.pkl") )
scaler_X = joblib.load(os.path.join(current_dir,"model/improvement_score/scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(current_dir,"model/improvement_score/scaler_y.pkl"))

def improvement_score(success_count, attempt_count, game_score_xp, game_level,engagement_time_Total_sec):
    # Prepare the user inputs as a numpy array
    user_inputs = np.array([success_count, attempt_count, game_score_xp, game_level,engagement_time_Total_sec]).reshape(1, -1)

    # Disable warnings
    warnings.filterwarnings("ignore")

    # Scale the user inputs using the loaded scalers
    user_inputs_scaled = scaler_X.transform(user_inputs)

    # Enable warnings again
    warnings.filterwarnings("default")

    # Make the prediction using the loaded model
    predicted_score_scaled =LR_model.predict(user_inputs_scaled)

    # Inverse transform to get the predicted score back to the original scale
    predicted_score = scaler_y.inverse_transform(predicted_score_scaled.reshape(1, -1))

   

    return  predicted_score[0][0] * 10


def find_cosine_similarity(y1, y2):
    point1=y1
    point2=y2
    # Calculate the dot product of the two vectors
    dot_product = sum(a * b for a, b in zip(point1, point2))

    # Calculate the magnitudes of the two vectors
    magnitude1 = math.sqrt(sum(a ** 2 for a in point1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in point2))

    # Calculate the cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    return 1-cosine_similarity

def arima(engagement_time_Total_sec,n_predictions):

  model = pm.auto_arima(engagement_time_Total_sec, seasonal=False, m=1, suppress_warnings=True)

  # Summary of the selected model
  #print(model.summary())

  n_periods = n_predictions # You can adjust this based on how many periods you want to forecast
  forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
  return forecast



def self_improvement_main(player_name:str,df_path:str,game_name=None):

  #Return json
  Real_time={}
  ID=""
  Future_weeks_prediction={}

  def real_time_improvement(last_attend_improvement_score , previouse_attend_improvement_score):
    if last_attend_improvement_score > previouse_attend_improvement_score:
      massage=f"This time you improve your workout completion rate by {improvement_presentage}% compared to the previous time."
      trend="Positive"
      return {"Massage":massage,"This_attend_improvement_score":last_attend_improvement_score,"previouse_attend_improvement_score":previouse_attend_improvement_score,
              "improvement_presentage":f"{improvement_presentage}%","trend":trend}

    elif last_attend_improvement_score < previouse_attend_improvement_score:
      massage=f"This time you decrease your exercise completion rate by {improvement_presentage}% compared to the previous time."
      trend="Negative"
      return {"Massage":massage,"This_attend_improvement_score":last_attend_improvement_score,"previouse_attend_improvement_score":previouse_attend_improvement_score,
              "improvement_presentage":f"{improvement_presentage}%","trend":trend}

    if last_attend_improvement_score == previouse_attend_improvement_score:
      massage=f"This time you not improve your workout completion rate compared to the previous time."
      trend="Normal"
      return {"Massage":massage,"This_attend_improvement_score":last_attend_improvement_score,"previouse_attend_improvement_score":previouse_attend_improvement_score,
              "improvement_presentage":f"{improvement_presentage}%","trend":trend}

  try:
    df=pd.read_csv(df_path)
    df = df.rename(columns={"engagement_time_mins": "engagement_time_Total_sec"})
    #df=fire_base_player_game(player_name)
  
    if len(df) != 0:
      #get the last attend game details
      pre_game=df.iloc[-1]
      #print(pre_game["game_name"])

      if game_name ==None:
        ##select palyer last game and level
        cal_df=df[(df["game_name"]==pre_game["game_name"])]
        cal_df = cal_df.copy()
      elif game_name !=None:
        cal_df=df[(df["game_name"]==game_name)]
        cal_df = cal_df.copy()
        #print(cal_df["game_name"][0])

      #create success_ratio & Success_engagement_time_mins columns & engagement_time_mins_Per_attend
      cal_df["success_ratio"]=(cal_df["success_count"] /cal_df["attempt_count"])
      cal_df["Success_engagement_time_sec"]=(cal_df["success_ratio"]*cal_df["engagement_time_Total_sec"])
      cal_df["engagement_time_sec_Per_attend"]=(cal_df["engagement_time_Total_sec"]/cal_df["attempt_count"])

      #set date colunm type to date
      #cal_df.loc[:, "date"] = pd.to_datetime(cal_df["date"])#, format="%d/%m/%Y"
      cal_df["date"] = pd.to_datetime(cal_df["date"])#, format="%d/%m/%Y"
      cal_df=cal_df.sort_values(by="date")#sort dataframe base by date





      if len(cal_df)==0:
        massage="There is not enough data to process"
        Real_time={"Massage":massage,"This_attend_improvement_score":None,"previouse_attend_improvement_score":None,"improvement_presentage":None,"trend":None}
        ID=0
        Future_weeks_prediction=None

      elif len(cal_df)==1:

        last_attend=cal_df.iloc[0]
        #last game improvement_score
        last_attend_improvement_score=improvement_score (success_count=last_attend["success_count"], attempt_count=last_attend["attempt_count"],
                                                        game_score_xp=last_attend["game_score_xp"], game_level=last_attend["game_level"],
                                                        engagement_time_Total_sec=last_attend["engagement_time_Total_sec"])
        #print(last_attend_improvement_score)
        massage=f"This time you improve your workout completion to {last_attend_improvement_score}"

        Real_time={"Massage":massage,"This_attend_improvement_score":last_attend_improvement_score
                  ,"previouse_attend_improvement_score":None,"improvement_presentage":None,"trend":None}
        ID=0
        Future_weeks_prediction=None

      elif len(cal_df) >= 2:


        #get the last and previouse attend details
        last_attend=cal_df.iloc[-1]
        previouse_attend=cal_df.iloc[-2]


        #last game improvement_score
        last_attend_improvement_score=improvement_score (success_count=last_attend["success_count"], 
                                                        attempt_count=last_attend["attempt_count"], 
                                                        game_score_xp=last_attend["game_score_xp"], 
                                                        game_level=last_attend["game_level"], 
                                                        engagement_time_Total_sec=last_attend["engagement_time_Total_sec"])

        #previouse game improvement_score
        previouse_attend_improvement_score=improvement_score (success_count=previouse_attend["success_count"], 
                                                              attempt_count=previouse_attend["attempt_count"], 
                                                              game_score_xp=previouse_attend["game_score_xp"],
                                                              game_level=previouse_attend["game_level"],
                                                              engagement_time_Total_sec=previouse_attend["engagement_time_Total_sec"])

        #compaired improvement
        y1=(last_attend["success_ratio"],last_attend["game_level"],last_attend["Success_engagement_time_sec"],
            last_attend["game_score_xp"])
        y2=(previouse_attend["success_ratio"],previouse_attend["game_level"],previouse_attend["Success_engagement_time_sec"],
            previouse_attend["game_score_xp"])

        improvement_presentage=find_cosine_similarity(y1, y2)
        improvement_presentage=round(improvement_presentage,2)

        #Real time predictions
        real_time=real_time_improvement(last_attend_improvement_score,previouse_attend_improvement_score)
        Real_time=real_time


        if len(cal_df) <=7:
          Real_time=real_time
          ID=1
          Future_weeks_prediction=None

        elif len(cal_df) <= 14:
          #get the last 14 game attended dates
          start_date=cal_df.iloc[-len(cal_df)]
          end_date=cal_df.iloc[-1]

          #create last 14 attend frequancy in day
          completion_freq=len(cal_df)/((end_date["date"]-start_date["date"]).days)

          #create ARIMA model to previous prediction
          engagement_time_mins=cal_df["engagement_time_sec_Per_attend"]

          number_predictions=int(np.round(7/completion_freq))
          n_predictions=''
          if number_predictions >=28:
            n_predictions=28
          else:
            n_predictions=number_predictions

          #predict nextweek engagement time
          forcast=arima(engagement_time_mins,int(n_predictions))
          after_week_engagement_time=forcast[len(cal_df)+int(n_predictions)-1]

          #predict nextweek improvment score
          next_week_improvement_score=improvement_score (success_count=cal_df["success_count"].mean(),
                                                        attempt_count=cal_df["attempt_count"].mean(),
                                                        game_score_xp=cal_df["game_score_xp"].mean(),
                                                        game_level=last_attend["game_level"],
                                                        engagement_time_Total_sec=after_week_engagement_time)


          if next_week_improvement_score >= last_attend_improvement_score:
            Future_weeks_prediction={"Improvment":"Positive","Completion frequency":f"{completion_freq} perday",
                                    "After_week_Success_engagements_Time_Min":after_week_engagement_time,
                                    "future_week_Success_improvement_score":next_week_improvement_score,"Predict_weeks":1}
            ID=2

          else :
            Future_weeks_prediction={"Improvment":"Normal","Completion frequency":f"{completion_freq} perday",
                                    "After_week_Success_engagements_Time_Min":after_week_engagement_time,
                                    "future_week_Success_improvement_score":next_week_improvement_score,"Predict_weeks":1}
            ID=2



        else:
          #get the last 7 game attended dates
          start_date=cal_df.iloc[-14]
          end_date=cal_df.iloc[-1]

          #create last 7 attend frequancy in day
          play_game_count=len(cal_df.iloc[-14:])
          completion_freq=play_game_count/((end_date["date"]-start_date["date"]).days)

          #crate ARIMA model to previous prediction
          engagement_time_mins=cal_df["engagement_time_sec_Per_attend"]

          number_predictions=int(np.round(14/completion_freq))
          n_predictions=''
          if number_predictions >=28:
            n_predictions=28
          else:
            n_predictions=number_predictions

          forcast=arima(engagement_time_mins,int(n_predictions))

          #predict nextweek engagement time
          after_week_engagement_time=forcast[len(cal_df)+int(n_predictions)-1]

          #predict nextweek improvment score
          next_week_improvement_score=improvement_score (success_count=cal_df["success_count"].mean(),
                                                        attempt_count=cal_df["attempt_count"].mean(), game_score_xp=cal_df["game_score_xp"].mean(),
                                                        game_level=last_attend["game_level"], engagement_time_Total_sec=after_week_engagement_time)


          if next_week_improvement_score >= last_attend_improvement_score:
            Future_weeks_prediction={"Improvment":"Positive","Completion frequency":f"{completion_freq} perday",
                                    "After_two_two_week_Success_engagements_Time_Min":after_week_engagement_time,
                                    "future_week_Success_improvement_score":next_week_improvement_score,"Predict_weeks":2}
            ID=2

          else :
            Future_weeks_prediction={"Improvment":"Normal","Completion frequency":f"{completion_freq} perday",
                                    "After_two_two_week_Success_engagements_Time_Min":after_week_engagement_time,
                                    "future_week_Success_improvement_score":next_week_improvement_score,"Predict_weeks":2}
            ID=2


    else:
      return {"Massage":"There is not enough data to process"}

    return {"ID":ID,"Real_Time_predictions":Real_time,'Future_weeks_Predictions':Future_weeks_prediction}


  except Exception as e:
    #print(e)
    return {"Massage":f"There is an error :{e} ,to process data"}