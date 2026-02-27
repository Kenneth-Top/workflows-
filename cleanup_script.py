
import os
import shutil

targets = [
    r"d:\Coding_project\python_work\Intern\LLM tokens\LLM_tokens_repo\test_fix.py",
    r"d:\Coding_project\python_work\Intern\LLM tokens\LLM_tokens_repo\test_fix_v2.py",
    r"d:\Coding_project\python_work\Intern\LLM tokens\LLM_tokens_repo\test_apis.py",
    r"d:\Coding_project\python_work\Intern\LLM tokens\LLM_tokens_repo\tmp_inspect.py",
    r"d:\Coding_project\python_work\Intern\LLM tokens\LLM_tokens_repo\debug_match.py",
]

for t in targets:
    if os.path.exists(t):
        try:
            os.remove(t)
            print(f"Deleted: {t}")
        except Exception as e:
            print(f"Error deleting {t}: {e}")
    else:
        print(f"Not found: {t}")

print("Cleanup script finished.")
