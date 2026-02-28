from bs4 import BeautifulSoup

with open("temp_arena_text.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
table = soup.find("table")
for r in table.find_all("tr")[1:3]:
    cells = r.find_all(['td', 'th'])
    print("Raw HTML of cell 2 (Model):")
    print(cells[2].prettify())
    print("-----")
