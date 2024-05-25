import pandas as pd
import json
import requests

data = pd.read_csv('data_client.csv')

# Convert DataFrame to list
data_list = data.values.tolist()

url = 'http://localhost:8000/predict/'

predictions = []
for record in data_list:
    payload = {'features': record}
    response = requests.post(url, json=payload)
    response.raise_for_status() 
    prediction = response.json()['prediction']
    predictions.append(prediction)

# Save to CSV
pd.DataFrame(predictions, columns=['Prediction']).to_csv('prediction_from_web.csv')
