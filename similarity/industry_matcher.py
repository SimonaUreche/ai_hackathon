def load_industries(path='config/industries.txt'):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

#momentan hardcode, urmeaza sa extragem din cv si jobs
INDUSTRIES = load_industries()

def compute_industry_score(cv_text, job_text):
    job_text = job_text.lower()
    cv_text = cv_text.lower()
    job_industries = [i for i in INDUSTRIES if i in job_text]
    matched = [i for i in job_industries if i in cv_text]

    if not job_industries:
        return 1.0
    if len(matched) == 0:
        return 0.0
    elif len(matched) < len(job_industries):
        return 0.5
    return 1.0
