import requests
import json

data = {
    'TV': 180.8,
    'Radio': 10.8,
    'Newspaper': 58.4,
}


url = 'https://advertising-ogkgw3gxza-uc.a.run.app/predict'

response = requests.post(url, json=data)
result = response.json()
print(json.dumps(result, indent=1))

