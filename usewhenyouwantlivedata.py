import requests
import json
def get_crypto_market_data_livecoinwatch():
  payload = json.dumps({
    "currency": "USD",
    "sort": "rank",
    "order": "ascending",
    "offset": 0,
    "limit": 50,
    "meta": True
  })

  headers = {
    'content-type': 'application/json',
    'x-api-key':"114153e3-9f9c-4842-8f0f-8a7cbe2d5680"
  }

  response = requests.request("POST", url, headers=headers, data=payload)

  # save response as a json file in data folder
  with open('data/crypto_market_data_livecoinwatch.json', 'w') as f:
    json.dump(response.json(), f)
  return response.json()
