from bs4 import BeautifulSoup

with open("temp_arena_text.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
tables = soup.find_all("table")
print(f"Found {len(tables)} tables")
if tables:
    for i, t in enumerate(tables):
        rows = t.find_all("tr")
        print(f"Table {i} has {len(rows)} rows.")
        if len(rows) > 10:
            # Print a few rows just to check
            for r in rows[1:5]:
                cells = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
                print(" | ".join(cells))
