# src/preprocessing/build_index.py

import os
import json
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer

from DB.session import engine, Base, SessionLocal
from DB.models  import CV

from src.preprocessing.extract_text    import load_docx_from_folder
from src.preprocessing.parse_skills    import parse_skills_from_description
from src.preprocessing.parse_industry  import parse_industry_from_description

PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH   = os.path.join(PROJECT_ROOT, "DataSet", "cv")
INDEX_PATH     = os.path.join(PROJECT_ROOT, "data", "cv_embeddings_hnsw.index")
DB_PATH        = os.path.join(PROJECT_ROOT, "data", "cvs_metadata.sqlite")
EMB_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    # ensure data/ exists
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    docs, filenames, descriptions = load_docx_from_folder(DATASET_PATH, is_cv=True)

    model = SentenceTransformer(EMB_MODEL_NAME)
    embs  = model.encode(docs, convert_to_numpy=True).astype("float32")

    dim          = embs.shape[1]
    num_elements = embs.shape[0]
    p = hnswlib.Index(space='cosine', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    p.add_items(embs, np.arange(num_elements))
    p.set_ef(50)
    p.save_index(INDEX_PATH)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    for fname, desc in zip(filenames, descriptions):
        skills          = parse_skills_from_description(desc)
        industry_scores = parse_industry_from_description(desc,skills)
        cv = CV(
            filename      = fname,
            skills_json   = json.dumps(skills, ensure_ascii=False),
            industry_json = json.dumps(industry_scores, ensure_ascii=False),
        )
        db.add(cv)
    db.commit()
    db.close()

    print("built successfully.")

if __name__ == "__main__":
    main()
