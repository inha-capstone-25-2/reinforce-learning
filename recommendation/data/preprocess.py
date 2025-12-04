import re
from typing import List

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower()

def tokenize_keywords(text: str) -> List[str]:
    text = normalize_text(text)
    text = re.sub(r"[^0-9a-zA-Z]+", " ", text)
    return [t for t in text.split() if t]