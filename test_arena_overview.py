import requests
from bs4 import BeautifulSoup

url = "https://arena.ai/leaderboard"
headers = {'User-Agent': 'Mozilla/5.0'}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.text, "html.parser")
table = soup.find("table")
if table:
    header_row = table.find("tr")
    headers = [c.get_text(strip=True) for c in header_row.find_all(['th', 'td'])]
    print("Overview Headers:", headers)

    for r in table.find_all("tr")[1:3]:
        cells = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
        print("Row:", cells)
