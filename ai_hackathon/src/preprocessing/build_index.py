# src/preprocessing/build_index.py

import os
from DB.session import engine, Base, SessionLocal
from DB.models  import JobDescription, JobIndustryScore
from src.preprocessing.parse_industry  import get_industry_scores_from_text, jd_prompt

PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH        = os.path.join(PROJECT_ROOT, "data", "cvs_metadata.sqlite")

def main():
    # ensure data/ exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    JOB_DESC_PATH = os.path.join(PROJECT_ROOT, "DataSet", "job_descriptions")
    job_files = [f for f in os.listdir(JOB_DESC_PATH) if f.endswith(".docx")]

    for idx, job_fname in enumerate(job_files):
        job_path = os.path.join(JOB_DESC_PATH, job_fname)
        print(f"Procesez Job Description {idx+1}/{len(job_files)}: {job_fname}")

        # Citește textul job description-ului
        from docx import Document
        doc = Document(job_path)
        job_text = "\n".join([para.text for para in doc.paragraphs])

        # Extrage industriile și explicațiile
        industry_scores, explanations = get_industry_scores_from_text(job_text, jd_prompt)

        # Creează și inserează JobDescription
        job = JobDescription(filename=job_fname, text=job_text)
        db.add(job)
        db.commit()  # ca să primească id

        # Inserează scorurile pe industrii pentru job
        for industry, score in industry_scores.items():
            explanation = explanations.get(industry, "")
            job_industry_score = JobIndustryScore(
                job_id=job.id,
                industry=industry,
                score=score,
                explanation=explanation
            )
            db.add(job_industry_score)
    db.commit()
    db.close()

    print("Datele pentru job descriptions au fost inserate în baza de date.")

if __name__ == "__main__":
    main()
