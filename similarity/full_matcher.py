from similarity.skill_matcher import compute_skill_score
from similarity.industry_matcher import compute_industry_score
from models.jobbert_matcher import compute_jobbert_similarity
from utils.explanations import generate_explanation


def compute_full_match(cv_text, job_text):
    #JobBERT (60%)
    jobbert_score = compute_jobbert_similarity(cv_text, job_text)

    #skilluri (TF-IDF, 30%)
    skill_score = compute_skill_score(cv_text, job_text)

    #industrie (keyword-based, 10%)
    industry_score = compute_industry_score(cv_text, job_text)

    final_score = (
        0.6 * jobbert_score +
        0.3 * skill_score +
        0.1 * industry_score
    )

    explanation = generate_explanation(jobbert_score, skill_score, industry_score)

    return round(final_score, 4), explanation
