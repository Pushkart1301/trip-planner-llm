import requests
import json

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "3 days in Mumbai"}
)
print(json.dumps(response.json(), indent=2))