# src/preprocessing/extract_text.py

import glob
import os
from docx import Document

def load_docx_from_folder(folder_path, is_cv=True):
    documents   = []
    filenames   = []
    descriptions = []

    for filepath in glob.glob(os.path.join(folder_path, '*.docx')):
        # sari peste fișierele temporare Word care încep cu "~$"
        fname = os.path.basename(filepath)
        if fname.startswith("~$"):
            continue

        # încarcă textul din .docx
        text = extract_text_from_docx(filepath=filepath, is_cv=is_cv)
        # curăță newline-urile și spațiile duble
        text = text.replace('\n', ' ').replace('  ', ' ')

        # separă descrierea relevantă
        if not is_cv:
            # pentru fișiere job descriptions
            text = text.split("Benefits:")[0]
            description = text.split("Key Responsibilities:")[0]
        else:
            # pentru CV-uri; preia doar secțiunea de proiecte
            parts = text.split("Project Experience")
            description = parts[1] if len(parts) > 1 else ""

        documents.append(text)
        filenames.append(fname)
        descriptions.append(description)

    return documents, filenames, descriptions


def extract_text_from_docx(filepath, is_cv):
    """
    Deschide un .docx și returnează textul din paragrafe, 
    sărind peste primul paragraf la CV-uri (header).
    """
    doc = Document(filepath)
    start_index = 1 if is_cv else 0
    return ' '.join(para.text for para in doc.paragraphs[start_index:])
