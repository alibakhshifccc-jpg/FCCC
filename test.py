import requests

url = "http://192.168.2.132:2030/"

response = requests.get(url)

print(response.status_code)