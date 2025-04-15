CV_EXTRACTION_PROMPT = """
You will receive a candidate's CV below. Please extract and return the following fields in JSON format:
- Full Name
- Technical Skills
- Programming Languages
- Tools & Technologies
- Industries they've worked in
- Estimated years of experience
- Short summary (2–3 lines)

CV:
\"\"\"
{cv_text}
\"\"\"
"""

JOB_EXTRACTION_PROMPT = """
You will receive a job description below. Please extract and return the following fields in JSON format:
- Job Title
- Required Technical Skills
- Relevant Industry
- Years of experience required
- Short summary

Job Description:
\"\"\"
{job_text}
\"\"\"
"""
