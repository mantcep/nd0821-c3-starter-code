from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_welcome():
    """Test that the root shows a welcome message."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to a demo ML web app!"


def test_infer_positive_pred():
    """Test positive prediction."""
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
    r = client.post('/infer', json=data)
    assert r.status_code == 200
    assert r.json()['prediction'] == 1


def test_infer_negative_pred():
    """Test negative prediction."""
    data = {
        "age": 27,
        "workclass": "Private",
        "fnlwgt": 113464,
        "education": "1st-4th",
        "education_num": 2,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Other",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 35,
        "native_country": "Dominican-Republic"
    }
    r = client.post('/infer', json=data)
    assert r.status_code == 200
    assert r.json()['prediction'] == 0
