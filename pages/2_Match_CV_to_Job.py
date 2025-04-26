import streamlit as st
import spacy
import glob
import os
import numpy as np
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
from utils import (
    load_docx_from_folder,
    extract_text_from_docx,
    spacy_tokenizer,
    progress_bar_update,
    nlp
)



# Set your folders here
cv_folder = './DataSet/cv'
job_folder = './DataSet/job_descriptions'

# Dicționar cu domenii industriale și cuvinte cheie asociate
# Folosit pentru calculul Industry Knowledge Score (10% din scorul total)
domain_data = { 
    "Banking": { "desc": "Developed software for banking platforms, digital wallets, loan processing, or credit systems.", 
                 "keywords": ["bank", "loan", "credit", "atm", "fintech", "interest", "account", "ledger"] }, 
    "Healthcare": { "desc": "Built medical systems such as EHR, hospital platforms, clinical apps, or telemedicine services.", 
                    "keywords": ["healthcare", "ehr", "patient", "hospital", "clinic", "medical", "doctor", "nurse"] }, 
    "E-commerce": { "desc": "Created platforms or tools for online stores, shopping carts, payments, or product discovery.", 
                    "keywords": ["ecommerce", "checkout", "cart", "payment", "shopify", "woocommerce", "product", "sku"] }, 
    "Telecommunications": { "desc": "Engineered tools for telecom networks, call management, VoIP, or network monitoring.", 
                            "keywords": ["telecom", "sms", "voip", "5g", "network", "bandwidth", "subscriber", "lte"] }, 
    "Education": { "desc": "Built education platforms such as learning portals, student dashboards, or LMS systems.", 
                   "keywords": ["education", "student", "teacher", "learning", "course", "classroom", "lms", "school"] }, 
    "Retail": { "desc": "Developed software for retail businesses such as POS systems, inventory, or loyalty programs.", 
                "keywords": ["retail", "store", "inventory", "pos", "stock", "sku", "receipt", "shopping"] }, 
    "Insurance": { "desc": "Created applications for policy management, claims processing, or underwriting systems.", 
                   "keywords": ["insurance", "claim", "policy", "underwriting", "premium", "broker", "risk"] }, 
    "Legal": { "desc": "Built tools for legal case management, document processing, or e-discovery platforms.", 
               "keywords": ["legal", "law", "contract", "case", "compliance", "jurisdiction", "litigation"] }, 
    "Manufacturing": { "desc": "Developed automation systems, supply chain tools, or MES solutions for production plants.", 
                       "keywords": ["manufacturing", "plant", "automation", "mes", "machine", "factory", "assembly"] }, 
    "Transportation & Logistics": { "desc": "Built platforms for delivery tracking, fleet management, logistics optimization, or routing.", 
                                    "keywords": ["logistics", "delivery", "routing", "fleet", "dispatch", "transport", "warehouse"] }, 
    "Energy & Utilities": { "desc": "Developed monitoring systems, SCADA platforms, or analytics for power and water utilities.", 
                            "keywords": ["energy", "power", "electricity", "gas", "grid", "meter", "solar", "utility", "scada"] }, 
    "Real Estate": { "desc": "Engineered platforms for property listings, CRM tools for agents, or real estate analytics.", 
                     "keywords": ["real estate", "property", "mortgage", "agent", "listing", "tenant", "lease"] }, 
    "Government": { "desc": "Built public service portals, civic data dashboards, or digital identity platforms.", 
                    "keywords": ["government", "municipal", "civic", "permit", "id", "citizen", "registry"] }, 
    "Marketing": { "desc": "Built marketing analytics tools, email campaign platforms, or digital ad performance systems.", 
                   "keywords": ["marketing", "campaign", "seo", "email", "ads", "promotion", "branding", "targeting"] }, 
    "Media & Entertainment": { "desc": "Created digital content platforms, streaming services, or entertainment production tools.", 
                               "keywords": ["media", "streaming", "video", "music", "entertainment", "broadcast", "subscriber"] }, 
    "Construction": { "desc": "Engineered project management tools, BIM integrations, or field apps for construction teams.", 
                      "keywords": ["construction", "site", "blueprint", "project", "bim", "architect", "contractor"] }, 
    "Finance (non-banking)": { "desc": "Worked on accounting systems, budgeting tools, payroll, or financial planning platforms.", 
                               "keywords": ["finance", "accounting", "budget", "payroll", "invoice", "expense", "audit", "report"] }
}

def extract_skills_from_cv(cv_text):
    """Extrage skill-uri din CV folosind NLP"""
    doc = nlp(cv_text)
    skills = set()
    
    # Caută substantive și fraze nominale care sunt probabil skill-uri
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:  # Ignoră frazele prea lungi
            skills.add(chunk.text.lower())
    
    # Verifică secțiuni speciale (Skills, Technical Skills etc.)
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

def get_matching_scores_between_cv_and_job_descriptions(cv_text, job_texts, progress_bar, status_text):
    """
    Calculează scorul de potrivire între un CV și mai multe job descriptions.
    Adaptare a funcției originale pentru match CV → Jobs.
    """
    # Preprocesăm CV-ul
    cv_text_preprocessed = " ".join(spacy_tokenizer(cv_text))
    
    # Preprocesăm job descriptions
    job_texts_preprocessed = [" ".join(spacy_tokenizer(job_text)) for job_text in job_texts]

    progress_bar_update(30, progress_bar, status_text)

    # Calculăm similaritatea semantică (spaCy embeddings)
    cv_doc = nlp(cv_text_preprocessed)
    similarities_embeddings = [
        cv_doc.similarity(nlp(job_text_preprocessed))
        for job_text_preprocessed in job_texts_preprocessed
    ]

    progress_bar_update(70, progress_bar, status_text)

    # Vectorizare TF-IDF
    combined_texts = [cv_text_preprocessed] + job_texts_preprocessed
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Separăm vectorii
    cv_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]

    # Similaritate cosinus
    similarities_tfidf = cosine_similarity(cv_vector, job_vectors)

    progress_bar_update(80, progress_bar, status_text)

    # Normalizare
    scaler = MinMaxScaler()
    similarities_tfidf_scaled = scaler.fit_transform(
        np.array(similarities_tfidf).reshape(-1, 1)
    ).flatten()

    progress_bar_update(90, progress_bar, status_text)

    # Combinăm scorurile (60% semantic, 40% lexical)
    similarities_embeddings = np.array(similarities_embeddings)
    final_scores = 0.6 * similarities_embeddings + 0.4 * similarities_tfidf_scaled

    return final_scores

st.title("📌 Match CV → Job")
progress_text = "Operation in progress. Please wait. ⏳"
uploaded_file = st.file_uploader("Upload a CV (.docx format)", type=["docx"])
progress_bar = st.progress(0)
status_text = st.empty()


start_button = st.button("Find Best Jobs for My CV")

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
    
     # Afișăm rezultatul - doar cel mai bun job
    st.subheader("🎯 Best Job Match")
    st.write(f"**Job Title:** {job_files[best_job_idx]}")
    st.write(f"**Match Score:** {final_scores[best_job_idx]*100:.1f}%")
    st.write(f"**General Match:** {general_match_scores[best_job_idx]*100:.1f}%")
    st.write(f"**Skills Match:** {skill_scores[best_job_idx]} common skills")
    if domain:
        st.write(f"**Domain Match:** {domain_scores[best_job_idx]} keywords")
    
    # Afișăm descrierea jobului
    st.subheader("Job Description")
    st.write(job_texts[best_job_idx])
    
    # Afișăm detaliile potrivirii
    st.subheader("🔍 Match Analysis")
    
    # Skill-uri comune
    common_skills = [
        skill for skill in skills 
        if re.search(rf'\b{re.escape(skill)}\b', job_texts[best_job_idx].lower())
    ]
    
    st.write("**Common Skills:**")
    for skill in common_skills[:10]:  # Afișăm primele 10 skill-uri comune
        st.write(f"- {skill}")
    
    # Keywords de domeniu comune (dacă există domeniu)
    if domain:
        common_domain_keywords = [
            kw for kw in domain_data[domain]["keywords"] 
            if kw in job_texts[best_job_idx].lower()
        ]
        
        st.write(f"**Common {domain} Keywords:**")
        for kw in common_domain_keywords:
            st.write(f"- {kw}")
    
    progress_bar_update(100, progress_bar, status_text)
    st.balloons()