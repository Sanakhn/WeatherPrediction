from django.shortcuts import render

# Create your views here.

#Basic imports 
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

API_KEY = '46a600d0468427c5fcacb416090dbb16'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

#getting data
def get_weather(city):
  url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
  response = requests.get(url)
  data = response.json()
  return {
      'city': data['name'],
      'current_temp': round(data['main']['temp']),
      'feels_like': round(data['main']['feels_like']),
      'temp_min': round(data['main']['temp_min']),
      'temp_max': round(data['main']['temp_max']),
      'humidity': round(data['main']['humidity']),
      'description': (data['weather'][0]['description']),
      'country': data['sys']['country'],
      'wind_gust_dir': data['wind']['deg'],
      'pressure': data['main']['pressure'],
      'Wind_Gust_Speed': data['wind']['speed']
  }


#getting historical data
def historical_data(filename):
  df = pd.read_csv(filename)
  df = df.dropna()
  df = df.drop_duplicates()

  return df


#preparing training data
def prepare_data(data):
  encoder = LabelEncoder()
  data['WindGustDir'] = encoder.fit_transform(data['WindGustDir'])
  data['RainTomorrow'] = encoder.fit_transform(data['RainTomorrow'])

  X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
  y = data['RainTomorrow']

  return X, y, encoder

#classification model
def train_classification_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  print("Mean Squared Error for Rain Model: ")
  print(mean_squared_error(y_test, y_pred))

  return model

#regression model
def regression_data(data, feature):
  X, y = [], []

  for i in range(len(data) - 1):
    X.append(data[feature].iloc[i])
    y.append(data[feature].iloc[i+1])

  X = np.array(X).reshape(-1, 1)
  y = np.array(y)
  return X, y

def train_regression_model(X, y):
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X, y)
  return model

#predicting future values
def predict(model, current_value):
  predictions = [current_value]

  for i in range(5):
    next_value = model.predict(np.array([[predictions[-1]]]))
    predictions.append(next_value[0])

  return predictions[1:]


#from django.http import HttpResponse

def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')

        current_weather = get_weather(city)

        # loading historical data
        csv_path = os.path.join('/Users/sana/weatherForecasting/weather.csv')
        data = historical_data(csv_path)

        X, y, encoder = prepare_data(data)

        # training the first model
        rain_model = train_classification_model(X, y)

        # map wind direction to compass points 
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("W", 258.75, 281.25), ("WW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]

        compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
        #encoding compass direction
        compass_direction_encoded = encoder.transform([compass_direction])[0] if compass_direction in encoder.classes_ else -1


        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['Wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }

        current_df = pd.DataFrame([current_data])

        # predicting for rain 
        rain_prediction = rain_model.predict(current_df)[0]

        # regression model for temp and humidity 
        X_temp, y_temp = regression_data(data, 'Temp')
        X_hum, y_hum = regression_data(data, 'Humidity')

        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        #predict future temperature and humidity using current

        future_temp = predict(temp_model, current_weather['temp_min'])
        future_humidity = predict(hum_model, current_weather['humidity'])

        #getting time for future prediction

        timezone = pytz.timezone('America/New_York')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

        future_time = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
        
        #store each value seperately

        time1, time2, time3, time4, time5 = future_time
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity

        #passing data to template 

        context = {
          'location': city,
          'current_temp': current_weather['current_temp'],
          'MinTemp': current_weather['temp_min'],
          'MaxTemp': current_weather['temp_max'],
          'feels_like': current_weather['feels_like'],
          'humidity': current_weather['humidity'],
          'city': current_weather['city'],
          'country': current_weather['country'], 

          'time': datetime.now(),
          'date': datetime.now().strftime("%B %d, %Y"), 

          'time1': time1,
          'time2': time2,
          'time3': time3,
          'time4': time4,
          'time5': time5,

          'temp1': f"{round(temp1, 1)}",
          'temp2': f"{round(temp2, 1)}",
          'temp3': f"{round(temp3, 1)}",
          'temp4': f"{round(temp4, 1)}",
          'temp5': f"{round(temp5, 1)}",

          'hum1': f"{round(hum1, 1)}",
          'hum2': f"{round(hum2, 1)}",
          'hum3': f"{round(hum3, 1)}",
          'hum4': f"{round(hum4, 1)}",
          'hum5': f"{round(hum5, 1)}",
        }

        return render(request, 'weather.html', context)
    
    return render(request, 'weather.html')