from datetime import datetime, timedelta
import os
from flask import jsonify
from data.database import DataBase
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from joblib import dump, load   
from xgboost import XGBRegressor

NUMBER_OF_COLUMNS = 16
SHARE_FOR_TRAINING = 0.85

class ModelCreator:
    def __init__(self):
        self.database = DataBase()
        self.modelPath = ''
        self.predicted_data = []
        self.predicted_date = None
        
    def create_sequences(data, n_steps):
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix > len(data)-1:
                break
            seq_x, seq_y = data[i:end_ix, :-1], data[end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def start_model_training(self, yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo):
        self.dataframe = self.load_data(yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo)

        # Create lag features (you can adjust the number of lags)
        
        df = self.dataframe

        scaler = MinMaxScaler()
        features = df.drop('Load', axis=1)
        target = df['Load']

        features_scaled = scaler.fit_transform(features)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

        # XGBoost Model
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

        # Train the model
        model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], verbose=False)

        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # # Define the filename with the timestamp
        filename = f"{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'models\\model_{current_timestamp}.joblib')}"
        
        dump(model, filename)

        print(f"Model saved as: {filename}")

    def predict(self, days, yearFrom, monthFrom, dayFrom, model_name):
        self.predicted_date = datetime(yearFrom, monthFrom, dayFrom)

        date_to = datetime(yearFrom, monthFrom, dayFrom) + timedelta(days=days - 1)
        self.dataframe = self.load_test_data(yearFrom, monthFrom, dayFrom, date_to.year, date_to.month, date_to.day) 
                                                                          #SHARE_FOR_TRAINING           
        
        X = self.dataframe.drop(['Load'], axis=1, errors='ignore')
        model = load(f"{os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'models/{model_name}')}")
        self.dataframe['Load'] = model.predict(X)
        
        columns_to_export = ['Hour', 'Day', 'Month', 'Year', 'Load']
        
        self.dataframe.to_csv("csv_filename.csv", index=False, columns=columns_to_export)
        
        self.dataframe['Timestamp'] = pd.to_datetime(self.dataframe[['Year', 'Month', 'Day', 'Hour']])
        
        dates = self.dataframe['Timestamp'].tolist()
        data = self.dataframe['Load'].tolist()
        
        return jsonify({"data": data, "dates": dates})

    def get_csv(self):
        if self.predicted_data == []:
            return {"error": "Error! No prediction!"}, 400
        
        rescaled_data, dates = self.scale_back()
        self.generate_csv(rescaled_data, dates)
        return {"data": "OK"}, 200

    def set_path(self, path):
        self.path = path

    def get_path(self):
        return os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'), self.path)

    def load_data(self, yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo):
        print("Load data started", datetime.now())
        dataframe = self.database.get_pandas_dataframe(yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo, True)
        print("Load data finished", datetime.now())
        return dataframe

    def load_test_data(self, yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo):
        print("Load data started", datetime.now())
        dataframe = self.database.get_pandas_dataframe(yearFrom, monthFrom, dayFrom, yearTo, monthTo, dayTo, False)
        print("Load data finished", datetime.now())
        return dataframe

    