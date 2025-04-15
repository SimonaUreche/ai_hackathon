import os
import nltk
from preprocessing import load_docx_from_folder
from similarity import compute_final_scores
from export_results import save_matrix_to_excel

nltk.download('stopwords')

# Configurații
cv_folder = 'DataSet/cv'
job_folder = 'DataSet/job_descriptions'
output_file = 'outputs/final_scores.xlsx'

# Încărcare date
cv_texts, cv_files = load_docx_from_folder(cv_folder)
job_texts, job_files = load_docx_from_folder(job_folder)

# Calcul scoruri (fără job_industries predefinite)
final_scores = compute_final_scores(cv_texts, job_texts)

# Afișare rezultate
for i, cv_file in enumerate(cv_files):
    print(f"\nCV: {cv_file}")
    for j, job_file in enumerate(job_files):
        print(f"  {job_file}: Score={final_scores[i][j]:.2f}")

# Salvare
os.makedirs(os.path.dirname(output_file), exist_ok=True)
save_matrix_to_excel(final_scores, cv_files, job_files, output_file)