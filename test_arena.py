import requests

url = "https://arena.ai/leaderboard/text"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}
resp = requests.get(url, headers=headers)
print(f"Status Code: {resp.status_code}")
print(f"Content Length: {len(resp.text)}")

with open("temp_arena_text.html", "w", encoding="utf-8") as f:
    f.write(resp.text)
print("Saved to temp_arena_text.html")
