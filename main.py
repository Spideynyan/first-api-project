import requests

url = "https://catfact.ninja/fact"

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()
    fact = data["fact"]

    print("Случайный факт о кошках:")
    print(fact)

except requests.RequestException as error:
    print("Ошибка при запросе к API:", error)
