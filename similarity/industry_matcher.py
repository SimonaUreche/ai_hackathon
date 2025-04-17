from sklearn.feature_extraction.text import TfidfVectorizer
import re


def extract_industries(text: str) -> list[str]:
    """
    Extract industries from text using common patterns and keywords.
    """
    # Common industry patterns with variations
    industry_patterns = [
        r'(?i)\b(tech(nology)?|IT|software)\b',
        r'(?i)\b(finance|banking|fintech)\b',
        r'(?i)\b(health(care)?|medical|pharma(ceutical)?)\b',
        r'(?i)\b(education|e-learning|teaching)\b',
        r'(?i)\b(retail|e-commerce)\b',
        r'(?i)\b(manufacturing|production)\b',
        r'(?i)\b(consulting|professional services)\b',
        r'(?i)\b(media|entertainment|gaming)\b',
        r'(?i)\b(telecom(munications)?)\b',
        r'(?i)\b(automotive|transportation)\b'
    ]
    
    industries = set()
    text = text.lower()
    
    # Extract industries using patterns
    for pattern in industry_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            # Clean and normalize the matched industry
            industry = match.group().strip()
            industries.add(industry)
    
    return list(industries)

def compute_industry_score(cv_text: str, job_text: str) -> float:
    """
    Compute industry match score between CV and job description.
    """
    job_industries = extract_industries(job_text)
    cv_industries = extract_industries(cv_text)
    
    if not job_industries:
        return 1.0  # No specific industry requirements
        
    matched = [i for i in job_industries if i in cv_industries]
    
    if len(matched) == 0:
        return 0.0
    elif len(matched) < len(job_industries):
        return 0.5
    return 1.0
