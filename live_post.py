import json
import requests

URL = 'https://demo-ml-webapp.herokuapp.com/infer/'
data = {
    "age": 42,
    "workclass": "Self-emp-inc",
    "fnlwgt": 152071,
    "education": "Prof-school",
    "education_num": 15,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 60,
    "native_country": "Cuba"
}

response = requests.post(URL, data=json.dumps(data))

print(f"Status code: {response.status_code}")
print(f"Prediction: {response.json()['prediction']}")
