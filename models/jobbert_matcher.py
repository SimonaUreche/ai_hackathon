# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F
#
# #JobBERT model
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # sau altul de pe Hugging Face
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)
#
# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]
#     return cls_embedding
#
# def compute_jobbert_similarity(cv_text, job_text):
#     emb_cv = get_embedding(cv_text)
#     emb_job = get_embedding(job_text)
#     similarity = F.cosine_similarity(emb_cv, emb_job).item()
#     similarity = (similarity + 1) / 2
#     return round(similarity, 4)

from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

def compute_jobbert_similarity(cv_text, job_text):
    emb_cv = sbert_model.encode(cv_text, convert_to_tensor=True)
    emb_job = sbert_model.encode(job_text, convert_to_tensor=True)
    score = util.cos_sim(emb_cv, emb_job).item()
    return round((score + 1) / 2, 4)
