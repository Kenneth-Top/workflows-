from bs4 import BeautifulSoup

with open("temp_arena_text.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
table = soup.find("table")
header_row = table.find("tr")
headers = [c.get_text(strip=True) for c in header_row.find_all(['th', 'td'])]
print("Headers:", headers)

for r in table.find_all("tr")[1:3]:
    cells = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
    print("Row:", cells)
