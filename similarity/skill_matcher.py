from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from typing import List
import re

def extract_skills(text: str) -> List[str]:
    """
    Extract skills from text using BERT-based token classification.
    Returns a list of identified skills.
    """
    # Initialize the NER pipeline with a BERT model
    classifier = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

    # Common technical skills patterns
    tech_patterns = [
        r'\b[Pp]ython\b', r'\b[Jj]ava\b', r'\b[Cc]\+\+\b', r'\b[Rr]eact\b',
        r'\b[Nn]ode\.?[Jj][Ss]\b', r'\b[Ss][Qq][Ll]\b', r'\b[Aa]gile\b',
        r'\b[Dd]ocker\b', r'\b[Kk]ubernetes\b', r'\b[Aa][Ww][Ss]\b',
        r'\b[Aa]zure\b', r'\b[Gg][Ii][Tt]\b', r'\b[Jj]ira\b'
    ]

    # Extract technical skills using regex
    tech_skills = set()
    for pattern in tech_patterns:
        matches = re.finditer(pattern, text)
        tech_skills.update(match.group().lower() for match in matches)

    # Use BERT to identify other potential skills
    try:
        entities = classifier(text)
        bert_skills = set(
            entity['word'].lower() for entity in entities
            if entity['score'] > 0.8  # High confidence threshold
        )
    except Exception:
        bert_skills = set()

    # Combine and clean skills
    all_skills = tech_skills.union(bert_skills)
    return list(all_skills)

def compute_skill_score(cv_text: str, job_text: str) -> float:
    """
    Compute similarity score between CV and job description based on skills.
    """
    # Extract skills from both texts
    cv_skills = extract_skills(cv_text)
    job_skills = extract_skills(job_text)

    if not cv_skills or not job_skills:
        return 0.0
    
    # Create a vocabulary from all identified skills
    all_skills = list(set(cv_skills + job_skills))
    
    # Use TF-IDF to compute similarity
    vectorizer = TfidfVectorizer(vocabulary=all_skills)
    tfidf = vectorizer.fit_transform([' '.join(cv_skills), ' '.join(job_skills)])
    score = (tfidf[0] @ tfidf[1].T).toarray()[0][0]
    return min(score, 1.0)
