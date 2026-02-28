import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def clean_model_name(raw_name_or_tag):
    if hasattr(raw_name_or_tag, 'find'):
        # On subpages, the model name is inside an <a> tag
        a_tag = raw_name_or_tag.find('a')
        if a_tag:
            name = a_tag.get_text(strip=True)
        else:
            name = raw_name_or_tag.get_text(strip=True)
    else:
        name = str(raw_name_or_tag).strip()
    
    # Strip company prefix from the name
    prefixes = ['Anthropic', 'Google', 'OpenAI', 'xAI', 'Meta', 'Mistral', 
                'Alibaba', 'DeepSeek', 'Zhipu', 'Baidu', 'ByteDance', 'Cohere',
                'Reka', 'AI21', 'Together', 'Nvidia', 'Amazon', 'Apple', 'Microsoft',
                'Tencent', 'Stability', 'Ideogram', 'Midjourney', 'Black Forest Labs',
                'Luma', 'Runway', 'Perplexity', 'You.com', 'Kling', 'Minimax',
                'Step', 'Moonshot', '01.AI', 'InternLM', 'Yi', 'Nous']
    for prefix in prefixes:
        if name.startswith(prefix) and len(name) > len(prefix):
            rest = name[len(prefix):]
            if rest[0].islower() or rest[0] in '-_.0123456789':
                name = rest
                break
    
    # Remove "/610" or similar trailing counts for Overview page
    name = re.sub(r'\d+/\d+$', '', name).strip()
    return name

def fetch_arena_leaderboard():
    from datetime import datetime
    all_data = {}
    today = datetime.now().strftime('%Y-%m-%d')
    headers = {'User-Agent': 'Mozilla/5.0'}

    OVERVIEW_COLS = ['Overall', 'Expert', 'Hard_Prompts', 'Coding', 
                     'Math', 'Creative_Writing', 'Instruction_Following', 'Longer_Query']
    
    # 1. Fetch Overview (for Ranks)
    print("Fetching Overview...")
    resp = requests.get('https://arena.ai/leaderboard', headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table')
    
    if len(tables) >= 9:
        overview_table = tables[8]
        for row in overview_table.find_all('tr')[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                model = clean_model_name(cells[0])
                if model not in all_data:
                    all_data[model] = {'Date': today, 'Model': model}
                
                for i, col in enumerate(OVERVIEW_COLS):
                    col_idx = i + 1
                    if col_idx < len(cells):
                        try:
                            rank_val = int(cells[col_idx].get_text(strip=True))
                            all_data[model][f'Rank_{col}'] = rank_val
                        except ValueError:
                            pass
    print(f"Overview Extracted: {len(all_data)} models")

    # 2. Fetch subpages for ELO scores
    LEADERBOARD_URLS = {
        'text': 'Score_text',
        'code': 'Score_code',
        'vision': 'Score_vision',
        'text-to-image': 'Score_text_to_image',
        'image-edit': 'Score_image_edit',
        'search': 'Score_search',
        'text-to-video': 'Score_text_to_video',
        'image-to-video': 'Score_image_to_video',
    }

    for path, col_name in LEADERBOARD_URLS.items():
        print(f"Fetching subpage /{path} ...")
        r = requests.get(f'https://arena.ai/leaderboard/{path}', headers=headers)
        if r.status_code != 200:
            print(f"Failed to fetch {path}")
            continue
        
        s = BeautifulSoup(r.text, 'html.parser')
        table = s.find('table')
        if not table:
            continue
        
        count = 0
        for row in table.find_all('tr')[1:]:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 5:
                # Based on /text: ['Rank', 'Rank Spread', 'Model', 'Score', 'Votes']
                model = clean_model_name(cells[2])
                
                score_str = cells[3].get_text(strip=True)
                score_match = re.search(r'^(\d+)', score_str)
                if score_match:
                    score = int(score_match.group(1))
                else:
                    score = None
                
                try:
                    votes = int(cells[4].get_text(strip=True).replace(',', ''))
                except ValueError:
                    votes = 0

                if score is not None:
                    if model not in all_data:
                        all_data[model] = {'Date': today, 'Model': model}
                    all_data[model][col_name] = score
                    all_data[model][f'Votes_{col_name.replace("Score_", "")}'] = votes
                    count += 1
        print(f"-> Found {count} scores for {col_name}")

    df = pd.DataFrame(list(all_data.values()))
    return df

if __name__ == "__main__":
    df = fetch_arena_leaderboard()
    print("\nData Preview:")
    print(df.head().to_string())
    print(f"\nTotal row count: {len(df)}")
    
    score_cols = [c for c in df.columns if c.startswith('Score_')]
    for c in score_cols:
        print(f"{c}: {df[c].notna().sum()} not null")
