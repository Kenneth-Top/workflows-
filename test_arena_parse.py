import json
from bs4 import BeautifulSoup

with open("temp_arena_text.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
script = soup.find("script", id="__NEXT_DATA__")
if script:
    data = json.loads(script.string)
    print("Found __NEXT_DATA__!")
    with open("temp_arena_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved to temp_arena_data.json (Check size/content)")
else:
    print("No __NEXT_DATA__ found.")
