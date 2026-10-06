import requests

url = "https://www.iiitn.ac.in/notices"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

res = requests.get(url, headers=headers)
print("Status Code:", res.status_code)
print("Content Length:", len(res.text))
print("Sample HTML:", res.text[:300])