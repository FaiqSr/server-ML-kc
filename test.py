import requests
import json
import time
import random # Import random

# Define the URL of your Flask endpoint
url = 'http://192.168.137.201:8001/predict' # Replace with your actual URL

# Prepare the data you want to send for prediction
# This should be a dictionary or a list of dictionaries
# with the keys matching the model's expected features:
# 'Temperature[C]', 'Humidity[%]', 'eCO2[ppm]'

# Convert the data to JSON format
headers = {'Content-Type': 'application/json'}

# Loop to send data every 2 seconds
while True:
    try:
        # Generate random dummy data within a reasonable range
        data_to_predict = {
            'Temperature[C]': [random.uniform(15.0, 50.0)], # Random temperature
            'Humidity[%]': [random.uniform(10.0, 60.0)],     # Random humidity
            'eCO2[ppm]': [random.randint(400, 2500)]       # Random eCO2
        }

        print(data_to_predict)

        # If you want to send multiple data points, use a list of dictionaries:
        # data_to_predict = [
        #     {'Temperature[C]': random.uniform(15.0, 50.0), 'Humidity[%]': random.uniform(10.0, 60.0), 'eCO2[ppm]': random.randint(400, 2500)},
        #     {'Temperature[C]': random.uniform(15.0, 50.0), 'Humidity[%]': random.uniform(10.0, 60.0), 'eCO2[ppm]': random.randint(400, 2500)},
        # ]


        # Send the POST request
        response = requests.post(url, data=json.dumps(data_to_predict), headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            predictions = response.json()
            print("Prediction received successfully:")
            print(predictions)
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError as e:
        print(f"Error: Could not connect to the server. Make sure your Flask app is running at {url}")
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Wait for 2 seconds before sending the next request
    time.sleep(2)