import pickle
import numpy as np

with open("models/ridge_model.pkl", "rb") as f:
    ridge_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_ridge(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = ridge_model.predict(scaled_input)
    return float(prediction[0])
