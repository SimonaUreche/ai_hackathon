import os
from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".docx":
        return extract_text_from_docx(path)
    elif ext == ".pdf":
        return extract_pdf_text(path)
    elif ext == ".txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
