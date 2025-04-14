# CV–Job Matching Engine (POC)

This project implements an intelligent CV–job matching pipeline, designed to evaluate how well a developer's CV matches a given job description.

The current version performs a full semantic and rule-based analysis using:

-  **Semantic similarity** with a sentence-transformers model (JobBERT)
-  **Technical skill matching** using TF-IDF based on a predefined skill list
-  **Industry relevance** through keyword-based matching

---

## ✅ What it does (current status)

- Loads CVs and job descriptions from `.docx` files
- Preprocesses and compares each CV to every job using:
  - Semantic vector similarity (60%)
  - Skill coverage from a predefined list (30%)
  - Industry keyword presence (10%)
- Generates a **final match score (0–1)** and a **textual explanation**
- Exports the results to an Excel file at:  
  `outputs/results.xlsx`

---

##  Sample Output 

Each row = one CV–job pair, with a match score and an explanation:

```csv
CV, Job, Score, Explanation
cv_100_Elena_Andreea_Vâlceanu.docx, job_description_100_UIUX Designer.docx, 0.5279, CV-ul conține experiență clară în industria căutată. CV-ul conține puține skilluri relevante. Textul CV-ului este bine aliniat cu descrierea jobului.
cv_100_Elena_Andreea_Vâlceanu.docx, job_description_10_Tech Lead.docx, 0.7104, CV-ul conține experiență clară în industria căutată. Doar o parte dintre skilluri sunt regăsite. Textul CV-ului este bine aliniat cu descrierea jobului.
cv_100_Elena_Andreea_Vâlceanu.docx, job_description_11_Product Owner.docx, 0.5214, CV-ul conține experiență clară în industria căutată. CV-ul conține puține skilluri relevante. Textul CV-ului este bine aliniat cu descrierea jobului.
cv_101_Anca_Miruna_Dobre.docx, job_description_100_UIUX Designer.docx, 0.5281, CV-ul conține experiență clară în industria căutată. CV-ul conține puține skilluri relevante. Textul CV-ului este bine aliniat cu descrierea jobului.
cv_101_Anca_Miruna_Dobre.docx, job_description_10_Tech Lead.docx, 0.6901, CV-ul conține experiență clară în industria căutată. Doar o parte dintre skilluri sunt regăsite. Textul CV-ului este bine aliniat cu descrierea jobului.
cv_101_Anca_Miruna_Dobre.docx, job_description_11_Product Owner.docx, 0.5430, CV-ul conține experiență clară în industria căutată. CV-ul conține puține skilluri relevante. Textul CV-ului este bine aliniat cu descrierea jobului.
cv_102_Andrei_Călin_Vasile.docx, job_description_100_UIUX Designer.docx, 0.5315, CV-ul conține experiență clară în industria căutată. CV-ul conține puține skilluri relevante. Textul CV-ului este bine aliniat cu descrierea jobului.
cv_102_Andrei_Călin_Vasile.docx, job_description_10_Tech Lead.docx, 0.7687, CV-ul conține experiență clară în industria căutată. Majoritatea skillurilor tehnice sunt prezente. Textul CV-ului este bine aliniat cu descrierea jobului.
cv_102_Andrei_Călin_Vasile.docx, job_description_11_Product Owner.docx, 0.5180, CV-ul conține experiență clară în industria căutată. CV-ul conține puține skilluri relevante. CV-ul are o potrivire parțială cu descrierea jobului.



