from bs4 import BeautifulSoup

with open("temp_arena_text.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
# Find links that look like leaderboard tabs
links = soup.find_all("a", href=True)
tabs = set()
for link in links:
    if "/leaderboard/" in link['href']:
        tabs.add(link['href'])

print("Found Leaderboard Tabs:", tabs)
