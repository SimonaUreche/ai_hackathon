import os
import nltk
from preprocessing import load_docx_from_folder
from similarity import compute_similarity_matrix
from export_results import save_matrix_to_excel

nltk.download('stopwords')

cv_folder = 'DataSet/cv'
job_folder = 'DataSet/job_descriptions'
output_file = 'outputs/similarity_results.xlsx'

cv_texts, cv_files = load_docx_from_folder(cv_folder)
job_texts, job_files = load_docx_from_folder(job_folder)

similarity = compute_similarity_matrix(cv_texts, job_texts)

for i, cv in enumerate(cv_files):
    print(f"\nSimilarity for {cv}:")
    for j, job in enumerate(job_files):
        print(f"  {job}: {similarity[i][j]:.4f}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
save_matrix_to_excel(similarity, cv_files, job_files, output_file)
