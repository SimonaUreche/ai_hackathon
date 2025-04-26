# src/preprocessing/parse_industry.py

from .domain_data import domain_data

def parse_industry_from_description(description: str, skills: list[str]) -> dict[str, float]:

    skills_set = set(tok.lower() for tok in skills)

    raw_scores = {}
    for domain, info in domain_data.items():
        kw_set = set(info["keywords"])
        matches = skills_set & kw_set
        # scor brut = număr potriviri / număr keyword-uri definite
        raw_scores[domain] = len(matches) / len(kw_set)

    # păstrăm doar cele cu scor > 0 și rotunjim
    industry_scores = {
        domain: round(score, 3)
        for domain, score in raw_scores.items()
        if score > 0
    }

    return industry_scores
