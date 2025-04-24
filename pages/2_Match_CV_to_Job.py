import streamlit as st
import spacy
import glob
import os
import numpy as np
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def load_docx_from_folder(folder_path, is_cv=True):
    documents = []
    filenames = []
    for filepath in glob.glob(os.path.join(folder_path, '*.docx')):
        text = extract_text_from_docx(filepath=filepath, is_cv=is_cv)
        text = text.replace('\n', ' ')
        text = text.replace('  ', ' ')
        if not is_cv:
            text = text.split("Benefits:")[0]
        documents.append(text)
        filenames.append(os.path.basename(filepath))
    return documents, filenames


def extract_text_from_docx(filepath, is_cv):
    doc = Document(filepath)
    return ' '.join([para.text for para in doc.paragraphs[1 if is_cv else 0:]])


# Custom tokenizer using spaCy
def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.text.isspace() and not token.text.isnumeric()]


# Set your folders here
cv_folder = './DataSet/cv'
job_folder = './DataSet/job_descriptions'


# Load documents
#cv_texts, cv_files = load_docx_from_folder(cv_folder)
#job_texts, job_files = load_docx_from_folder(job_folder, is_cv=False)



st.title("📌 Match CV → Job")
progress_text = "Operation in progress. Please wait. ⏳"
uploaded_file = st.file_uploader("Upload a CV (.docx format)", type=["docx"])
progress_bar = st.progress(0)
status_text = st.empty()