from sklearn.feature_extraction.text import TfidfVectorizer

def load_skills(path='config/skills.txt'):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

#momentan hardcode, urmeaza sa extragem din cv si jobs
SKILLS = load_skills()

def compute_skill_score(cv_text, job_text):
    vectorizer = TfidfVectorizer(vocabulary=SKILLS)
    tfidf = vectorizer.fit_transform([cv_text, job_text])
    score = (tfidf[0] @ tfidf[1].T).toarray()[0][0]
    return min(score, 1.0)
