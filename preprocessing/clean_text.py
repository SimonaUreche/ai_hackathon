import os
import glob
import string
from docx import Document
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def extract_text_from_docx(filepath):
    doc = Document(filepath)
    return '\n'.join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_docx_from_folder(folder_path):
    documents = []
    filenames = []
    for filepath in glob.glob(os.path.join(folder_path, '*.docx')):
        text = extract_text_from_docx(filepath)
        documents.append(preprocess_text(text))
        filenames.append(os.path.basename(filepath))
    return documents, filenames
