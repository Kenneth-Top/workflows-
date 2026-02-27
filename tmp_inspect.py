"""Inspect the HTML structure to understand how to parse leaderboard data"""
from bs4 import BeautifulSoup

html = open('lmarena_raw.html', 'r', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')

h3s = soup.find_all('h3')
# Index 1 = 文本排行榜 (the first leaderboard with scores)
target_h3 = h3s[1]
print(f"H3 text: {target_h3.text}")

# Find the next sibling that contains list items
next_elem = target_h3.find_next_sibling()
while next_elem and next_elem.name not in ['ol', 'ul', 'div', 'h3']:
    next_elem = next_elem.find_next_sibling()

if next_elem:
    print(f"\nNext sibling tag: <{next_elem.name}> class={next_elem.get('class')}")
    print(f"Content preview: {str(next_elem)[:500]}")

# Try a different approach: find the parent div of h3 and look for list items there
parent = target_h3.parent
print(f"\nParent tag: <{parent.name}> class={parent.get('class')}")

# Find all li within the parent
lis = parent.find_all('li')
print(f"\nLI items in parent: {len(lis)}")
for li in lis[:5]:
    print(f"  LI text: {repr(li.text)}")
    print(f"  LI html: {str(li)[:200]}")

# Try a broader approach: find the ranking-card div that contains this h3
ranking_card = target_h3.find_parent('div', class_=True)
while ranking_card:
    lis_in_card = ranking_card.find_all('li')
    if len(lis_in_card) > 3:
        print(f"\nFound container: <{ranking_card.name}> class={ranking_card.get('class')}")
        print(f"  Contains {len(lis_in_card)} LI items")
        for li in lis_in_card[:3]:
            print(f"  LI text: {repr(li.text)}")
            print(f"  LI html: {str(li)[:300]}")
        break
    ranking_card = ranking_card.find_parent('div', class_=True)
    
# Also: just find ALL ordered lists
ols = soup.find_all('ol')
print(f"\n\nTotal <ol> tags: {len(ols)}")
for i, ol in enumerate(ols[:3]):
    lis = ol.find_all('li')
    print(f"\n  OL[{i}]: {len(lis)} items, class={ol.get('class')}")
    for li in lis[:3]:
        print(f"    LI text: {repr(li.text)}")
