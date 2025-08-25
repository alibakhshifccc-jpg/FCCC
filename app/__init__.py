import requests

auth_url = "http://192.168.2.132:2030/swagger/ui/index#!/About/About_GetDllVersion"

auth_res = requests.get(auth_url)

print(auth_res.status_code)

for i in auth_res:
    print(i)