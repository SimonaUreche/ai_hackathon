import streamlit as st
import numpy as np
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from utils.common_functions import (load_docx_from_folder,spacy_tokenizer,progress_bar_update, nlp, domain_data)
import re
from utils.explanation import generate_explanation_with_llm_cv_to_job
from concurrent.futures import ThreadPoolExecutor

cv_folder = './DataSet/cv'
job_folder = './DataSet/job_descriptions'

def extract_skills_from_cv(cv_text):
    """Extrage skill-uri din CV folosind NLP"""
    doc = nlp(cv_text)
    skills = set()


    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:
            skills.add(chunk.text.lower())

    skills_section = re.search(r'(?:skills|technical skills|competencies):(.*?)(?:\n\n|\n\w+:|$)',
                               cv_text, re.IGNORECASE)
    if skills_section:
        for token in nlp(skills_section.group(1)):
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                skills.add(token.text.lower())

    return list(skills)


def detect_domain_from_cv(cv_text, domain_data):
    """Detectează domeniul din CV folosind dicționarul de keywords"""
    cv_text_lower = cv_text.lower()
    domain_scores = {}  # Inițializăm dicționarul gol

    for domain, data in domain_data.items():
        domain_scores[domain] = 0  # Inițializăm scorul pentru fiecare domeniu
        for keyword in data["keywords"]:
            if keyword.lower() in cv_text_lower:
                domain_scores[domain] += 1

    if not domain_scores:
        return None

    # Returnăm domeniul cu cele mai multe potriviri
    return max(domain_scores.items(), key=lambda x: x[1])[0]


from concurrent.futures import ThreadPoolExecutor

def get_matching_scores_between_cv_and_job_descriptions(cv_text, job_texts, progress_bar, status_text):
    """
    Calculează scorul de potrivire între un CV și mai multe job descriptions, paralelizat pe batchuri.
    """
    def compute_batch_similarity(batch, cv_doc):
        return [cv_doc.similarity(nlp(job_text)) for job_text in batch]

    def chunk_list(lst, n):
        return [lst[i:i+n] for i in range(0, len(lst), n)]

    # Preprocesare
    cv_text_preprocessed = " ".join(spacy_tokenizer(cv_text))
    job_texts_preprocessed = [" ".join(spacy_tokenizer(job_text)) for job_text in job_texts]

    progress_bar_update(30, progress_bar, status_text)

    # Embedding pentru CV
    cv_doc = nlp(cv_text_preprocessed)

    # Batch-uim job descriptions
    batch_size = 50
    chunks = chunk_list(job_texts_preprocessed, batch_size)
    num_threads = min(8, len(chunks))

    # Paralelizare cu ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda batch: compute_batch_similarity(batch, cv_doc), chunks)

    # Combinăm toate scorurile de similaritate semantică
    similarities_embeddings = [sim for batch in results for sim in batch]

    progress_bar_update(70, progress_bar, status_text)

    # TF-IDF similarity
    combined_texts = [cv_text_preprocessed] + job_texts_preprocessed
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    cv_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]

    similarities_tfidf = cosine_similarity(cv_vector, job_vectors)

    progress_bar_update(80, progress_bar, status_text)

    # Normalizare TF-IDF
    scaler = MinMaxScaler()
    similarities_tfidf_scaled = scaler.fit_transform(
        np.array(similarities_tfidf).reshape(-1, 1)
    ).flatten()

    progress_bar_update(90, progress_bar, status_text)

    # Combinăm scorurile
    similarities_embeddings = np.array(similarities_embeddings)
    final_scores = 0.6 * similarities_embeddings + 0.4 * similarities_tfidf_scaled

    return final_scores



st.title("📌 Match CV → Job")
progress_text = "Operation in progress. Please wait. ⏳"
uploaded_file = st.file_uploader("Upload a CV", type=["docx"])
progress_bar = st.progress(0)
status_text = st.empty()


col1, col2, col3 = st.columns([1,2,1])
with col2:
    start_button = st.button(
        "Find Best Jobs for My CV",
        type="primary",
        use_container_width=True
    )
if start_button:
    if not uploaded_file:
        st.error("❌ Please upload a job description file")


if start_button and uploaded_file:
    # Procesăm CV-ul încărcat
    doc = Document(uploaded_file)
    cv_text = ' '.join([para.text for para in doc.paragraphs])
    cv_text = cv_text.replace('\n', ' ').replace('  ', ' ')

    progress_bar_update(10, progress_bar, status_text)

    # Extragem skill-uri și domeniu din CV
    skills = extract_skills_from_cv(cv_text)
    domain = detect_domain_from_cv(cv_text, domain_data)

    # Încărcăm job descriptions
    job_texts, job_files, _ = load_docx_from_folder(job_folder, is_cv=False)

    progress_bar_update(30, progress_bar, status_text)

    # Calculăm scorurile de potrivire
    general_match_scores = get_matching_scores_between_cv_and_job_descriptions(
        cv_text, job_texts, progress_bar, status_text
    )

    # Calculăm scoruri pentru skill-uri
    skill_scores = np.array([
        sum(1 for skill in skills if re.search(rf'\b{re.escape(skill)}\b', job_text.lower()))
        for job_text in job_texts
    ])

    # Calculăm scoruri pentru domeniu (dacă a fost detectat)
    if domain:
        domain_keywords = domain_data[domain]["keywords"]
        domain_scores = np.array([
            sum(1 for kw in domain_keywords if kw in job_text.lower())
            for job_text in job_texts
        ])
    else:
        domain_scores = np.zeros(len(job_texts))

    progress_bar_update(70, progress_bar, status_text)

    # Normalizăm scorurile
    scaler = MinMaxScaler()
    skill_scores_norm = scaler.fit_transform(skill_scores.reshape(-1, 1)).flatten()
    domain_scores_norm = scaler.fit_transform(domain_scores.reshape(-1, 1)).flatten()

    # Calculăm scorul final (40% potrivire generală, 35% skill-uri, 25% domeniu)
    final_scores = (
            0.4 * general_match_scores +
            0.35 * skill_scores_norm +
            0.25 * domain_scores_norm
    )

    best_job_idx = np.argmax(final_scores)

    progress_bar_update(90, progress_bar, status_text)


    st.subheader("🎯 Best Job Match")
    st.write(f"*Job Title:* {job_files[best_job_idx]}")
    st.write(f"*Match Score:* {final_scores[best_job_idx] * 100:.1f}%")
    st.write(f"*General Match:* {general_match_scores[best_job_idx] * 100:.1f}%")
    st.write(f"*Skills Match:* {skill_scores[best_job_idx]} common skills")
    if domain:
        st.write(f"*Domain Match:* {domain_scores[best_job_idx]} keywords")

    st.subheader("The top matching Job is:")
    st.write(job_texts[best_job_idx])

    common_skills = [
        skill for skill in skills
        if re.search(rf'\b{re.escape(skill)}\b', job_texts[best_job_idx].lower())
    ]

    st.header("Selection Explanation:")
    explanation_text = generate_explanation_with_llm_cv_to_job(
        cv_filename="Uploaded CV",
        domain_score=domain_scores_norm[best_job_idx],
        skills_score=skill_scores_norm[best_job_idx],
        matching_score=general_match_scores[best_job_idx],
        matched_skills=common_skills,
        domain_selected=domain
    )

    st.subheader("Explanation for the Best Match")
    st.write(explanation_text)

    progress_bar_update(100, progress_bar, status_text)
    st.balloons()