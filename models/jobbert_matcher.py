from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# Load the model once
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

def compute_jobbert_similarity(cv_text, job_text):
    """
    Compute semantic similarity between CV and job description.
    Returns normalized score between 0 and 1.
    """
    # Encode texts (make sure they are lists or strings)
    emb_cv = model.encode(cv_text, convert_to_tensor=True)
    emb_job = model.encode(job_text, convert_to_tensor=True)

    # Compute cosine similarity
    score = util.cos_sim(emb_cv, emb_job).item()

    # Normalize score from [-1, 1] to [0, 1]
    normalized_score = (score + 1) / 2
    return round(normalized_score, 4)

def get_top_matches(job_text, all_cv_texts, top_k=5):
    """
    Find top matching CVs for a given job description.
    Returns list of tuples (index, score) sorted by score descending.
    """
    # Encode job and CVs
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    cv_embeddings = model.encode(all_cv_texts, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(job_embedding, cv_embeddings)[0]  # 1D tensor of similarity scores

    # Convert similarities to numpy for argsort
    similarities_np = similarities.cpu().numpy()
    top_indices = np.argsort(similarities_np)[-top_k:][::-1]  # Highest first

    # Format output
    top_scores = [round(float(similarities[i]), 4) for i in top_indices]
    return list(zip(top_indices, top_scores))
