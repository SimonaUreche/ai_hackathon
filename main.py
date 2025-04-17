import os
import pandas as pd
from preprocessing.clean_text import load_docx_from_folder, clean_text
from similarity.full_matcher import compute_full_match

cv_folder = "DataSet/cv"
job_folder = "DataSet/job_descriptions"
output_file = "outputs/results.xlsx"

cv_texts, cv_files = load_docx_from_folder(cv_folder)
job_texts, job_files = load_docx_from_folder(job_folder)

cv_texts, cv_files = cv_texts[:3], cv_files[:3]
job_texts, job_files = job_texts[:3], job_files[:3]

cv_texts = [clean_text(text) for text in cv_texts]
job_texts = [clean_text(text) for text in job_texts]

results = []

for i, cv_text in enumerate(cv_texts):
    for j, job_text in enumerate(job_texts):
        final_score, explanation = compute_full_match(cv_text, job_text)

        results.append({
            "CV": cv_files[i],
            "Job": job_files[j],
            "Score": final_score,
            "Explanation": explanation
        })

print(results)
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_excel(output_file, index=False)
