
import re

def clean_text(text: str) -> str:
    """
    Clean raw text extracted from documents:
    - remove extra whitespace
    - strip control characters
    - normalize formatting
    """
    text = re.sub(r"\s+", " ", text)  # reduce multiple whitespaces to one
    text = text.replace('\x0c', '')   # remove form feed (from PDFs)
    return text.strip()
