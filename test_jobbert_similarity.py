import os
import json
from preprocessing.file_converter import extract_text
from preprocessing.llm_cv_parser import parse_cv_with_llm
from preprocessing.llm_job_parser import parse_job_with_llm
from models.jobbert_matcher import compute_jobbert_similarity

# 1. Căi către fișiere
cv_folder = "DataSet/cv"
job_path = "DataSet/job_descriptions/job_description_32_Full Stack Developer.docx" 

# 2. Extract + Parse job
print("🔍 Extracting and parsing job description...")
job_text = extract_text(job_path)
job_data = parse_job_with_llm(job_text)

# Convertim dict-ul într-un text simplu
job_full_text = json.dumps(job_data, ensure_ascii=False)

# 3. Procesăm toate CV-urile
cv_scores = []
print("\n📄 Processing CVs...")

for filename in os.listdir(cv_folder):
    if filename.endswith(".docx"):
        cv_path = os.path.join(cv_folder, filename)
        try:
            cv_text = extract_text(cv_path)
            cv_data = parse_cv_with_llm(cv_text)
            cv_full_text = json.dumps(cv_data, ensure_ascii=False)

            score = compute_jobbert_similarity(cv_full_text, job_full_text)
            cv_scores.append((filename, score))
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# 4. Afișăm top 5
print("\n🏆 Top 5 CV-uri potrivite:")
cv_scores.sort(key=lambda x: x[1], reverse=True)

for i, (filename, score) in enumerate(cv_scores[:5], 1):
    print(f"{i}. {filename} — Scor: {score:.4f}")
