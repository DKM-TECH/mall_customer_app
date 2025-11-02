import pickle
import numpy as np

with open("models/linear_model.pkl", "rb") as f:
    linear_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_linear(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = linear_model.predict(scaled_input)
    return float(prediction[0])
