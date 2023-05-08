import requests

def request(url, inputs):
    results = requests.post(url, data=inputs)

    return results.json()
