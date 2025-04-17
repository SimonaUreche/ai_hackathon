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
job_data = parse_job_with_llm(job_text)  # Folosim direct structura parsată

# 3. Procesăm toate CV-urile
cv_scores = []
print("\n📄 Processing CVs...")

for filename in os.listdir(cv_folder):
    if filename.endswith(".docx"):
        cv_path = os.path.join(cv_folder, filename)
        try:
            cv_text = extract_text(cv_path)
            cv_data = parse_cv_with_llm(cv_text)  # Folosim direct structura parsată
            
            # Calculăm scorul cu ambele versiuni de text (raw + parsate)
            score_raw = compute_jobbert_similarity(cv_text, job_text)  # Doar text brut
            score_enhanced = compute_jobbert_similarity(
                cv_text, 
                job_text,
                cv_parsed=cv_data,
                job_parsed=job_data
            )
            
            cv_scores.append((filename, score_raw, score_enhanced))
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# 4. Afișăm top 5 pentru ambele metode
def print_top(scores, title, col_idx):
    print(f"\n🏆 {title}:")
    scores.sort(key=lambda x: x[col_idx], reverse=True)
    for i, (filename, *scores) in enumerate(scores[:5], 1):
        print(f"{i}. {filename} — Scor: {scores[col_idx-1]:.4f}")

print_top(cv_scores, "Top 5 CV-uri (text brut)", 1)
print_top(cv_scores, "Top 5 CV-uri (cu date structurate)", 2)