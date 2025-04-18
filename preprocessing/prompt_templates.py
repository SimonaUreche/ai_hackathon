CV_EXTRACTION_PROMPT = '''
You will receive a candidate's CV in raw text format. Your task is to extract and return a JSON object with the following fields:

- "name": Full name of the candidate.
- "skills": All relevant hard and soft skills mentioned.
- "technologies": Tools, platforms, or technologies the candidate has worked with (e.g., Python, ReactJS, Docker, Figma).
- "industries": Industries or fields inferred from projects, tools, and context (e.g., Retail, Healthcare, Fitness).
- "experience_years": Estimated number of years of full-time professional work experience. Do NOT include university years unless the candidate held full-time jobs during that time. Internships or student projects do not count unless explicitly stated as work experience.
- "relevant_experience": Number of years working specifically in technical fields (e.g., software engineering, machine learning, data science). You may include internships or part-time work **only if clearly described as technical experience**. Academic degrees or project work alone should not be counted as years of experience.
- "skill_ratings": Dictionary of skill → rating (1–5), inferred from context.
- "languages": Foreign languages and their proficiency levels (e.g., English: C1).
- "certifications": List of certifications the candidate has obtained.
- "education": Education background, structured by degree level, with institution and duration.
- "projects": A list of key projects. For each, include:
    - "title": Project title or focus
    - "technologies": Technologies used
    - "summary": 1–2 lines summarizing the project’s goal and candidate’s role
- "summary": A short 2–3 line professional summary based on the candidate's background and experience level.

IMPORTANT:
- Use your best judgment to estimate relevant experience based on project descriptions, tools used, and context.
- Return only a valid JSON object. Do not include explanations or extra commentary.

CV:
"""
{cv_text}
"""
'''


JOB_EXTRACTION_PROMPT = '''
You will receive a job description in plain text. Your task is to extract and return a JSON object with the following fields:

- "job_title": Official job title.
- "industry": The domain or sector of the job (e.g., IT, Finance, Healthcare, Education, Retail, etc.).
- "required_skills": Technical and soft skills that are mandatory or strongly desired.
- "preferred_skills": Skills or experience marked as optional or nice-to-have.
- "years_of_experience_required": Minimum required experience (e.g., "2+ years", "Mid-level").
- "technologies": Specific tools, platforms, frameworks, or languages mentioned (e.g., Java, AWS, Docker, React).
- "soft_skills": Any non-technical skills such as communication, leadership, teamwork.
- "summary": A 2–3 line summary of the job, highlighting responsibilities and team context.

IMPORTANT:
- Distinguish clearly between required and preferred skills.
- Return only a valid JSON object. Do not include explanations, markdown, or extra text.

Job Description:
"""
{job_text}
"""
'''
