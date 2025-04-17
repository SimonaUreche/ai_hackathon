
CV_EXTRACTION_PROMPT = '''
You will receive a candidate's CV in raw text format. Your task is to extract and return a JSON object with the following fields:

- "name": Full name of the candidate
- "skills": All relevant hard and soft skills mentioned (e.g., teamwork, budgeting, data analysis, leadership, customer service, etc.)
- "technologies": Any tools, platforms, systems or technologies the candidate has worked with. This can include software (e.g., Excel, SAP), hardware (e.g., lab equipment, CNC machines), or industry-specific platforms (e.g., CRM systems, medical imaging tools).
- "industries": List of industries or fields the candidate has worked in (e.g., Healthcare, Education, Retail, Logistics, Construction, IT, Finance, etc.). Infer from context if not explicitly mentioned.
- "experience_years": Estimated total years of relevant professional experience.
- "summary": A short (2–3 line) professional summary of the candidate’s profile. 
  Use a realistic tone that matches their level of experience. 
  If the candidate has less than 2 years of experience, use terms like "developing experience", 
  "early-stage", or "gaining expertise" instead of "skilled" or "experienced".

Return only a valid JSON matching the expected structure.

CV:
"""
{cv_text}
"""
'''

JOB_EXTRACTION_PROMPT = '''
You will receive a job description in plain text. Your task is to extract and return a JSON object with the following fields:

- "job_title": Official job title
- "required_skills": List of required technical and soft skills
- "industry": The domain or sector of the job (e.g., Finance, IT, Healthcare, Education, Retail, etc.)
- "years_of_experience_required": Minimum experience required (e.g., "2+ years", "Senior level", etc.)
- "summary": A 2–3 line summary of the job highlighting its main responsibilities and context.

Return only a valid JSON matching the expected structure.

Job Description:
"""
{job_text}
"""
'''
