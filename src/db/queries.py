import sqlite3
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_job_id_by_filename(db_path: str, job_filename: str) -> int:
    """Returnează ID-ul unui job description pe baza numelui fișierului"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM job_description WHERE filename = ?", (job_filename,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            logger.warning(f"Nu s-a găsit job cu numele fișierului: {job_filename}")
            return None
    except Exception as e:
        logger.error(f"Eroare la căutarea job-ului: {str(e)}")
        raise

def get_job_industry(db_path: str, job_id: int) -> str:
    """Returnează industria asociată unui job ID"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT industry FROM job_industry_score WHERE job_id = ? ORDER BY score DESC LIMIT 1",
            (job_id,)
        )
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            logger.warning(f"Nu s-a găsit industrie pentru job-ul cu ID: {job_id}")
            return None
    except Exception as e:
        logger.error(f"Eroare la căutarea industriei: {str(e)}")
        raise

def get_cv_industries(db_path: str, cv_filename: str) -> list:
    """Returnează industriile asociate unui CV pe baza numelui fișierului"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM cv WHERE filename = ?", (cv_filename,))
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Nu s-a găsit CV cu numele fișierului: {cv_filename}")
            return []
        cv_id = row[0]

        cursor.execute(
            "SELECT industry FROM cv_industry_score WHERE cv_id = ?",
            (cv_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        industries = [row[0] for row in rows]
        return industries
    except Exception as e:
        logger.error(f"Eroare la căutarea industriilor pentru CV-ul {cv_filename}: {str(e)}")
        raise



def get_cv_industry_scores(db_path: str, cv_filenames: list, selected_industry: str) -> np.ndarray:
    """Returnează un array de scoruri pentru fiecare CV față de industria selectată"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        scores = []

        for filename in cv_filenames:
            cursor.execute("SELECT id FROM cv WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Nu s-a găsit CV cu numele fișierului: {filename}")
                scores.append(0)
                continue
            cv_id = row[0]

            cursor.execute(
                "SELECT score FROM cv_industry_score WHERE cv_id = ? AND LOWER(industry) = LOWER(?)",
                (cv_id, selected_industry)
            )
            row = cursor.fetchone()
            if row:
                scores.append(row[0] / 100)  # normalizat 0-1
            else:
                logger.warning(f"Nu s-a găsit scor pentru CV-ul {filename} în industria {selected_industry}")
                scores.append(0)

        conn.close()
        return np.array(scores)
    except Exception as e:
        logger.error(f"Eroare la obținerea scorurilor: {str(e)}")
        raise


def get_job_industry_scores(db_path: str, job_filenames: list, selected_industry: str) -> np.ndarray:
    """Returnează un array de scoruri pentru fiecare job față de industria selectată"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        scores = []

        for filename in job_filenames:
            cursor.execute("SELECT id FROM job_description WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Nu s-a găsit job cu numele fișierului: {filename}")
                scores.append(0)
                continue
            job_id = row[0]

            # Obține scorul pentru industria selectată
            cursor.execute(
                "SELECT score FROM job_industry_score WHERE job_id = ? AND LOWER(industry) = LOWER(?)",
                (job_id, selected_industry)
            )
            row = cursor.fetchone()
            if row:
                scores.append(row[0] / 100)  # Normalizează scorul la intervalul [0, 1]
            else:
                logger.warning(f"Nu s-a găsit scor pentru job-ul {filename} în industria {selected_industry}")
                scores.append(0)

        conn.close()
        return np.array(scores)
    except Exception as e:
        logger.error(f"Eroare la căutarea scorurilor pentru joburile din industria {selected_industry}: {str(e)}")
        raise
