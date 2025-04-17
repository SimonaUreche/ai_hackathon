from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List, Tuple

# Load the model once
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

def compute_jobbert_similarity(cv_text: str, job_text: str, 
                              cv_parsed: Dict = None, 
                              job_parsed: Dict = None) -> float:
    """
    Compute semantic similarity between CV and job description with enhanced features.
    Uses both raw text and parsed structured data (if available).
    
    Args:
        cv_text: Raw CV text
        job_text: Raw job description text
        cv_parsed: Parsed CV data from LLM (optional)
        job_parsed: Parsed job data from LLM (optional)
    
    Returns:
        Normalized score between 0 and 1
    """
    # Base embeddings from raw text
    emb_cv = model.encode(cv_text, convert_to_tensor=True)
    emb_job = model.encode(job_text, convert_to_tensor=True)
    base_score = util.cos_sim(emb_cv, emb_job).item()

    # If parsed data is available, compute additional features
    if cv_parsed and job_parsed:
        # 1. Skills/Technologies match (weighted)
        cv_skills = set(cv_parsed.get("technologies", []) + 
                       [s.strip() for s in cv_parsed.get("skills", "").split(",")])
        job_skills = set(job_parsed.get("required_skills", []))
        
        skill_overlap = len(cv_skills & job_skills) / max(1, len(job_skills))
        
        # 2. Industry match (binary)
        cv_industries = set(cv_parsed.get("industries", []))
        job_industry = job_parsed.get("industry", "")
        industry_match = 1.0 if job_industry in cv_industries else 0.0
        
        # 3. Experience level (heuristic)
        try:
            cv_exp = float(cv_parsed.get("experience_years", "0").split("+")[0])
            job_exp = float(job_parsed.get("years_of_experience_required", "0").split("+")[0])
            exp_match = min(cv_exp / max(job_exp, 1), 1.0)  # Cap at 1.0
        except:
            exp_match = 0.5  # Default if parsing fails

        # Combined score (adjust weights as needed)
        enhanced_score = (
            0.6 * base_score + 
            0.25 * skill_overlap +
            0.1 * industry_match +
            0.05 * exp_match
        )
        return round(enhanced_score, 4)
    
    # Fallback to base score if no parsed data
    return round((base_score + 1) / 2, 4)  # Normalized to [0, 1]

def get_top_matches(job_text: str, all_cv_texts: List[str], 
                   all_cv_parsed: List[Dict] = None,
                   job_parsed: Dict = None, 
                   top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Find top matching CVs for a given job description with enhanced features.
    
    Args:
        job_text: Raw job description text
        all_cv_texts: List of raw CV texts
        all_cv_parsed: List of parsed CV data (optional)
        job_parsed: Parsed job data (optional)
        top_k: Number of top matches to return
    
    Returns:
        List of tuples (index, score) sorted by score descending
    """
    # Encode job and CVs
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    cv_embeddings = model.encode(all_cv_texts, convert_to_tensor=True)
    
    # Base similarities
    similarities = util.cos_sim(job_embedding, cv_embeddings)[0].cpu().numpy()
    
    # Enhance scores with parsed data if available
    if job_parsed and all_cv_parsed and len(all_cv_parsed) == len(all_cv_texts):
        enhanced_scores = []
        for i, (cv_text, cv_data) in enumerate(zip(all_cv_texts, all_cv_parsed)):
            enhanced_score = compute_jobbert_similarity(
                cv_text, job_text, cv_parsed=cv_data, job_parsed=job_parsed
            )
            enhanced_scores.append(enhanced_score)
        scores = np.array(enhanced_scores)
    else:
        scores = (similarities + 1) / 2  # Normalize to [0, 1]
    
    # Get top indices
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(i, round(float(scores[i]), 4)) for i in top_indices]