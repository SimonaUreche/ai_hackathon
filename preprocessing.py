import os
import glob
import spacy
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Încarcă modelul spaCy o singură dată
nlp = spacy.load("en_core_web_sm")  # Folosește modelul mic pentru viteză

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
    text_lower = text.lower()
    industries = {
        "banking": ["bank", "financial", "loan"],
        "healthcare": ["medical", "hospital", "health"],
        "IT": ["software", "tech", "developer"]
    }
    
    for industry, keywords in industries.items():
        if any(keyword in text_lower for keyword in keywords):
            return industry
    return "general"

def extract_skills(text):
    """Extrage abilitățile tehnice din text"""
    doc = nlp(text.lower())
    predefined_skills = ["python", "java", "react", "sql", "javascript", 
                        "html", "css", "c++", "machine learning", "flask"]
    return list(set(token.text for token in doc if token.text in predefined_skills))

def compute_industry_score(cv_text, job_industry):
    """Calculează scorul de potrivire a industriei"""
    cv_industry = extract_industry(cv_text)
    return 1.0 if cv_industry == job_industry else 0.0

def compute_technical_score(cv_text, job_text, technical_weights=None):
    """Calculează scorul tehnic (cu sau fără ponderi)."""
    if technical_weights:
        cv_skills = extract_skills(cv_text)
        total_score = sum(weight for skill, weight in technical_weights.items() 
                          if skill.lower() in [s.lower() for s in cv_skills])
        return total_score / 100  # Normalizare la [0, 1]
    else:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([cv_text, job_text])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]