
from sentence_transformers import SentenceTransformer, util

# SBERT model 
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity_matrix(cvs, jobs):
    cv_embeddings = sbert_model.encode(cvs, convert_to_tensor=True)
    job_embeddings = sbert_model.encode(jobs, convert_to_tensor=True)
    similarity_matrix = util.cos_sim(cv_embeddings, job_embeddings).cpu().numpy()
    return similarity_matrix
