import requests


url = r"https://diogohs-udacity-ml-devops.herokuapp.com/predict"


if __name__ == "__main__":
    request_data = {
        "age": 60,
        "workclass": "Local-gov",
        "fnlgt": 98350,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Other-service",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "Philippines"
    }

    response = requests.post(url, json=request_data)
    print(f"Status code: {response.status_code}")
    print(f"Return: {response.json()}")
