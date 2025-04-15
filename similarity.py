from sentence_transformers import SentenceTransformer, util
from preprocessing import compute_industry_score, compute_technical_score, extract_industry
import torch

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(cvs, jobs):
    """Calculează similaritatea semantică între CV-uri și job-uri"""
    if not cvs or not jobs:
        raise ValueError("Listele de CV-uri sau job-uri nu pot fi goale.")
    
    # Encodare batch pentru eficiență
    cv_embeddings = sbert_model.encode(cvs, convert_to_tensor=True)
    job_embeddings = sbert_model.encode(jobs, convert_to_tensor=True)
    
    # Asigură dimensiuni corecte
    if cv_embeddings.dim() == 1:
        cv_embeddings = cv_embeddings.unsqueeze(0)
    if job_embeddings.dim() == 1:
        job_embeddings = job_embeddings.unsqueeze(0)
    
    return util.cos_sim(cv_embeddings, job_embeddings).cpu().numpy()

def compute_final_scores(cvs, jobs, technical_weights=None):
    """Calculează scorurile finale combinate"""
    try:
        semantic_scores = compute_semantic_similarity(cvs, jobs)
        final_scores = []
        
        for i, cv_text in enumerate(cvs):
            row = []
            for j, job_text in enumerate(jobs):
                industry = extract_industry(job_text)
                industry_score = compute_industry_score(cv_text, industry)
                tech_score = compute_technical_score(cv_text, job_text, technical_weights)
                
                # Combinează scorurile cu ponderile specificate
                combined_score = (0.6 * semantic_scores[i][j] + 
                                0.3 * tech_score + 
                                0.1 * industry_score)
                row.append(combined_score)
            
            final_scores.append(row)
        
        return final_scores
        
    except Exception as e:
        raise RuntimeError(f"Eroare la calculul scorurilor finale: {str(e)}")