import requests
import dotenv
import os
import sys

sys.path.append("..")
dotenv.load_dotenv()
apikey = os.getenv("exchange_token")

def value_in_rupees(value, origin_currency):
    url = "https://currency-conversion-and-exchange-rates.p.rapidapi.com/convert"

    querystring = {"from":f"{origin_currency}","to":"INR","amount":f"{value}"}

    headers = {
        "x-rapidapi-key": apikey,
        "x-rapidapi-host": "currency-conversion-and-exchange-rates.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    print(response.json())
    return response.json()["result"]