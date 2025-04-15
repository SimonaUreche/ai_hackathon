import os
import glob
import spacy
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inițializare spaCy
nlp = spacy.load("en_core_web_lg")

def load_docx_from_folder(folder_path):
    """Încarcă toate fișierele .docx dintr-un folder"""
    documents = []
    filenames = []
    for filepath in glob.glob(os.path.join(folder_path, '*.docx')):
        doc = Document(filepath)
        text = '\n'.join([para.text for para in doc.paragraphs])
        documents.append(text)
        filenames.append(os.path.basename(filepath))
    return documents, filenames

def extract_industry(text):
    """Extrage industria din text"""
    doc = nlp(text.lower())
    industries = {
        "banking": ["bank", "financial", "loan"],
        "healthcare": ["medical", "hospital", "health"],
        "IT": ["software", "tech", "developer"]
    }
    
    for industry, keywords in industries.items():
        if any(keyword in doc.text for keyword in keywords):
            return industry
    return "general"

def compute_industry_score(cv_text, job_industry):
    """Calculează scorul de potrivire a industriei (0-1)"""
    cv_industry = extract_industry(cv_text)
    return 1.0 if cv_industry == job_industry else 0.0

def compute_technical_score(cv_text, job_text):
    """Calculează scorul tehnic folosind TF-IDF și cosine similarity"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cv_text, job_text])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]