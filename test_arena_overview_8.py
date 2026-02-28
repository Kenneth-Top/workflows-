import requests
from bs4 import BeautifulSoup

url = "https://arena.ai/leaderboard"
headers = {'User-Agent': 'Mozilla/5.0'}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, "html.parser")
tables = soup.find_all("table")
print(f"Total tables: {len(tables)}")

if len(tables) > 8:
    overview_table = tables[8]
    header_row = overview_table.find("tr")
    headers = [c.get_text(strip=True) for c in header_row.find_all(['th', 'td'])]
    print("Overview Headers:", headers)

    for r in overview_table.find_all("tr")[1:3]:
        cells = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
        print("Row:", cells)
