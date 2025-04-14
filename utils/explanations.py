def generate_explanation(jobbert_score, skill_score, industry_score):
    parts = []

    #Industry
    if industry_score == 1.0:
        parts.append("CV-ul conține experiență clară în industria căutată.")
    elif industry_score == 0.5:
        parts.append("CV-ul menționează parțial industria relevantă.")
    else:
        parts.append("CV-ul nu arată experiență în industria căutată.")

    #Skills
    if skill_score > 0.7:
        parts.append("Majoritatea skillurilor tehnice sunt prezente.")
    elif skill_score > 0.3:
        parts.append("Doar o parte dintre skilluri sunt regăsite.")
    else:
        parts.append("CV-ul conține puține skilluri relevante.")

    #Semantic Match
    if jobbert_score > 0.7:
        parts.append("Textul CV-ului este bine aliniat cu descrierea jobului.")
    elif jobbert_score > 0.4:
        parts.append("CV-ul are o potrivire parțială cu descrierea jobului.")
    else:
        parts.append("CV-ul se potrivește slab cu descrierea jobului.")

    return " ".join(parts)
