import pickle
import numpy as np

with open("models/lasso_model.pkl", "rb") as f:
    lasso_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_lasso(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = lasso_model.predict(scaled_input)
    return float(prediction[0])
