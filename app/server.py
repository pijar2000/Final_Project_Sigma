from fastapi import FastAPI
import joblib
import numpy as np

model, optimal_thresh = joblib.load('app/best_rf_model.joblib')

class_names = np.array(['0', '1'])  # '0' is for on-time and '1' is for late

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Shipping Model API'}

@app.post('/predict')
def predict(data: dict):
    # Extract features from the request
    features = np.array(data['features'])

    # Perform prediction
    prediction = model.predict(features.reshape(1, -1))
    predicted_class = class_names[prediction][0]

    return {'prediction': predicted_class}
