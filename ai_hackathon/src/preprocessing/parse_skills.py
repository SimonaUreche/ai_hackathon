import re
from typing import List, Dict
from src.preprocessing.domain_data import domain_data  

def parse_skills_from_description(description: str) -> List[str]:
    text = description.lower()
    # 1. Construim set-ul global de keyword-uri
    all_keywords = set()
    for info in domain_data.values():
        all_keywords.update(info["keywords"])

    # 2. Tokenizare
    tokens = re.findall(r"\b[\w\+\#]+\b", text)

    # 3. Filtrare și deduplicare păstrând ordinea
    seen = set()
    skills = []
    for tok in tokens:
        if tok in all_keywords and tok not in seen:
            seen.add(tok)
            skills.append(tok)
    return skills
