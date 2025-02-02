import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data

# Initialize the scaler
def init_scaler(data):
    """
    Initialize and fit the MinMaxScaler on the dataset.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    target = data['sales'].values.reshape(-1, 1)
    scaler.fit(target)
    return scaler

# Load the trained model
def load_trained_model(model_path):
    """
    Load the trained LSTM model from a file.
    """
    model = load_model(model_path)
    return model

# Preprocess input data for prediction
def preprocess_input(data, scaler, seq_length):
    """
    Preprocess the input data for the LSTM model.
    """
    sequence = data[-seq_length:]  # Use the last `seq_length` values
    sequence_scaled = scaler.transform(sequence.reshape(-1, 1))
    return sequence_scaled.reshape(1, seq_length, 1)

# Postprocess model output
def postprocess_output(prediction, scaler):
    """
    Reverse the scaling of the model's output.
    """
    return scaler.inverse_transform(prediction)