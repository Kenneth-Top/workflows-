import re

with open("temp_arena_text.html", "r", encoding="utf-8") as f:
    text = f.read()

# 找含有minimax-m2.5的片段
matches = [m.start() for m in re.finditer(r'minimax-m2\.5', text)]
print(f"Found {len(matches)} occurrences of 'minimax-m2.5'")

if matches:
    idx = matches[0]
    # 取前后 500 个字符看下数据结构
    snippet = text[max(0, idx-500):min(len(text), idx+500)]
    print("\n--- SNIPPET ---")
    print(snippet)
    print("----------------")
