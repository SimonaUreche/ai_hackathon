from sentence_transformers import SentenceTransformer, util
from preprocessing import compute_industry_score, compute_technical_score, extract_industry

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(cvs, jobs):
    """Calculează similaritatea semantică între CV-uri și job-uri"""
    cv_embeddings = sbert_model.encode(cvs, convert_to_tensor=True)
    job_embeddings = sbert_model.encode(jobs, convert_to_tensor=True)
    return util.cos_sim(cv_embeddings, job_embeddings).cpu().numpy()

def compute_final_scores(cvs, jobs):
    """
    Calculează scorurile finale combinând:
    - 60% similaritate semantică
    - 30% abilități tehnice
    - 10% potrivire industrie
    """
    # Calculează toate similaritățile semantice odată (optimizare)
    semantic_scores = compute_semantic_similarity(cvs, jobs)
    
    final_scores = []
    for i, cv_text in enumerate(cvs):
        row = []
        for j, job_text in enumerate(jobs):
            # Extrage industria automat din descrierea jobului
            current_job_industry = extract_industry(job_text)
            
            industry_score = compute_industry_score(cv_text, current_job_industry)
            tech_score = compute_technical_score(cv_text, job_text)
            
            row.append(0.6 * semantic_scores[i][j] + 
                      0.3 * tech_score + 
                      0.1 * industry_score)
        final_scores.append(row)
    
    return final_scores