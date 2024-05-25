import csv
import json
import requests

data = [[4, 177, 3, 0, 44, 1233],
        [4, 273, 3, 1, 2, 4374],
        [5, 180, 2, 0, 40, 1240],
        [4, 200, 5, 1, 50, 1200],
        [5, 245, 3, 0, 9, 4694],
        [6, 190, 4, 1, 48, 1250],
        [2, 150, 2, 0, 38, 1210],
        [4, 270, 3, 0, 1, 4131],
        [5, 167, 2, 1, 2, 5908],
        [3, 170, 2, 1, 43, 3245]]

url = 'http://localhost:8000/predict/'

predictions = []
for record in data:
    payload = {'features': record}
    response = requests.post(url, json=payload)
    prediction = response.json()['prediction']
    predictions.append(prediction)

# save to CSV
with open('prediction_web.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['prediction'])
    csvwriter.writerows([[pred] for pred in predictions])

print(predictions)
