import os
import glob
from docx import Document

def load_docx_from_folder(folder_path, is_cv=True):
    documents = [] #full text of the file
    filenames = [] #file names
    descriptions = [] #an important part of each file
    for filepath in glob.glob(os.path.join(folder_path, '*.docx')): #browse all files
        text = extract_text_from_docx(filepath=filepath, is_cv=is_cv) #is_cv/is_job
        text = text.replace('\n', ' ') #clean extracted text
        text = text.replace('  ', ' ')
        if not is_cv:
            text = text.split("Benefits:")[0]
            description = text.split("Key Responsibilities:")[0] #extract the description - everything before Key Responsibilities
        else:
            description = text.split("Project Experience")[1] #if it's a CV, extract the description as the part after Project Experience

        documents.append(text) #add the entire processed document
        filenames.append(os.path.basename(filepath)) #file name for final display
        descriptions.append(description) #previously extracted fragment

    return documents, filenames, descriptions


def extract_text_from_docx(filepath, is_cv):
    doc = Document(filepath)
    #if it's a CV start from paragraph 1 (ignore the name), if it's a job start from paragraph 0
    return ' '.join([para.text for para in doc.paragraphs[1 if is_cv else 0:]])

