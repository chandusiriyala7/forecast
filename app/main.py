from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime
from .schemas import QueryDate

# Load the model and scaler
model = load_model("/Users/siriyalachandu/Desktop/forecast/models/lstm_model.h5")
scaler = MinMaxScaler(feature_range=(0, 1))
data = pd.read_csv("/Users/siriyalachandu/Desktop/forecast/data/company_data.csv")
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
target = data['sales'].values.reshape(-1, 1)
scaler.fit(target)

# Initialize FastAPI
app = FastAPI()

@app.post("/predict")
def predict(query: QueryDate):
    try:
        query_date = datetime.strptime(query.date, "%Y-%m-%d")
        if query_date in data.index:
            idx = data.index.get_loc(query_date)
            if idx >= 30:
                sequence = data['sales'].values[idx-30:idx]
                sequence_scaled = scaler.transform(sequence.reshape(-1, 1))
                prediction = model.predict(sequence_scaled.reshape(1, 30, 1))
                prediction = scaler.inverse_transform(prediction)
                return {"date": query.date, "sales": float(prediction[0][0])}
            else:
                raise HTTPException(status_code=400, detail="Not enough data to predict for this date.")
        else:
            last_sequence = data['sales'].values[-30:]
            last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
            days_ahead = (query_date - data.index[-1]).days
            if days_ahead <= 0:
                raise HTTPException(status_code=400, detail="Date is in the past.")
            future_predictions = []
            for _ in range(days_ahead):
                next_pred = model.predict(last_sequence_scaled.reshape(1, 30, 1))
                future_predictions.append(next_pred[0, 0])
                last_sequence_scaled = np.append(last_sequence_scaled[1:], next_pred, axis=0)
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            return {"date": query.date, "sales": float(future_predictions[-1][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))