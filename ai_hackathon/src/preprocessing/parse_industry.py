import os
import openai
from dotenv import load_dotenv
from openai import OpenAI
import re
from docx import Document
# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

cv_prompt = """
You are an AI system that analyzes CVs to determine in which industries a person has had direct or tangential professional experience.

Instructions:
- Identify and output ALL industries in which the candidate has direct or tangential experience, not just those in the example.
- If the CV mentions any project, tool, data, or collaboration even remotely related to an industry (e.g., Retail, Fitness), assign at least 5% for that industry, even if the connection is weak or only tangential.
- First, output ONLY the list, one per line, in the format: Industry – Score%
- Then, after a blank line, output explanations for each industry, one per line, in the format: Industry: explanation (3 words max).
- Even if the candidate has no experience in any industry, output the list with at least one line (e.g., IT – 100% or IT – 0%).

Scoring Rules:
- 100%: ONLY if the candidate was employed directly in an industry-specific company AND had a non-IT core role (e.g., Retail Salesperson, Hospital Nurse, Banking Clerk).
- 50%: Only one mention of direct industry experience, in a non-technical context.
- 10%: Built a technical (IT) solution for that industry, but NOT employed directly in that industry.
- 5%: Peripheral or tangential exposure (e.g., mentioned industry data, collaborated with industry teams, design work for apps related to the industry).
- 0%: No verifiable connection.

Important:
- **Never assign 100%** for an industry based only on technical project work unless it is IT.
- **ALWAYS assign 100% IT** if the candidate has strong technical skills (Python, AWS, TensorFlow, SQL, JavaScript, React, etc.) AND led technical projects (e.g., developed platforms, built apps, deployed models) even if no formal Work Experience is listed.
- Having technical skills plus technical project leadership equals IT – 100%.
- Do not lower IT score due to lack of formal employment titles if projects demonstrate sufficient technical leadership.

ONLY consider:
- "Work Experience" or "Employment" sections
- OR "Project Experience" sections if they use active leadership verbs like "Led," "Managed," "Oversaw," "Acted as," "Served as."

IGNORE:
- Simple task descriptions ("developed app for retail") unless framed with leadership responsibility.

Examples of core tasks:
- Retail: inventory management, cashier, sales, merchandising
- Healthcare: patient care, clinical treatment
- Banking: teller operations, credit evaluation
- Education: teaching, mentorship, curriculum development
- IT: software development, data engineering, cloud infrastructure, ML model development, technical consulting

###

Analyze the following CV and generate the output following the rules above.

CV:
{text}
"""

jd_prompt = """
You are an AI system that analyzes job descriptions to identify the MAJOR INDUSTRIES in which a candidate is expected to have prior work experience.

Important Clarification:
- Only focus on MAJOR INDUSTRIES like IT, Retail, Healthcare, Banking, Education, Logistics, Manufacturing, Construction, Energy, Media, Entertainment, Government, Legal, Insurance, Telecommunications, Automotive, Fitness, Real Estate, Consulting, etc.
- DO NOT include roles, subfields, technical fields, skills, project types, tools, technologies, methodologies, or job functions like "Software Development", "Cloud Computing", "Agile Development", "DevOps", "Scrum", "Finance", "Human Resources", "Project Management", "Data Analysis", etc.
- ONLY INDUSTRY names must appear in the output.

Instructions:
- Ignore 'Benefits' or any section not related to the direct work/task that the employee will be doing at the job
- Identify and output ALL relevant industries, even if the connection is weak or only tangential.
- First, output ONLY the list, one per line, in the format: Industry – Score%
- Then, after a blank line, output explanations for each industry, one per line, in the format: Industry: explanation (3 words max).
- Even if no industries seem relevant, output at least one line

ONLY consider:
- "Job Title" which has the greatest importance
- "Work Experience", "Employment History", "Project Experience", "Key Responsibilities", "Preferred Skills", "Required Qualifications", and "Company Overview" sections.


Examples of core tasks:
- Retail: managing store operations, optimizing sales processes
- Healthcare: working in clinical projects, supporting patient systems
- Banking: fintech platform development, compliance projects
- Education: curriculum development, e-learning deployment
- IT: software development, cloud platform management
- Manufacturing: production line optimization, automation systems
- Logistics: warehouse management, supply chain systems
- Construction: infrastructure projects, civil engineering

###

Analyze the following doc and generate the output following the rules above.

Job description:
{text}
"""


def read_docx(file_path):
    """Citește tot textul dintr-un fișier .docx."""
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def get_industry_scores_from_text(text: str, prompt: str) -> tuple[dict, dict]:
    """
    Trimite textul (CV sau Job Description) și promptul către OpenAI și returnează:
    - un dicționar cu scorurile pe industrii
    - un dicționar cu explicațiile pentru fiecare industrie
    """
    full_prompt = prompt.format(text=text)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        max_tokens=900,
        temperature=0
    )
    content = response.choices[0].message.content

    #print("Full answer:\n\n\n", content)

    industry_scores = {}
    explanations = {}

    for line in content.splitlines():
        line = line.replace("–", "-").replace("—", "-").strip()
        line = re.sub(r"\s+", " ", line)
        match_score = re.match(r"(.+?)\s*-\s*([0-9]{1,3})\s*%", line)
        match_expl = re.match(r"(.+?):\s*(.+)", line)

        if match_score:
            industry = match_score.group(1).strip()
            score = int(match_score.group(2))
            industry_scores[industry] = score
        elif match_expl:
            industry = match_expl.group(1).strip()
            explanation = match_expl.group(2).strip()
            explanations[industry] = explanation
        #elif line:
            #print("Linie neprocesată ca scor:", repr(line))

    return industry_scores, explanations

def parse_industry(file_path: str, prompt: str) -> tuple[dict, dict]:
    """
    Citește un fișier .docx (CV sau Job Description) și returnează scorurile folosind OpenAI.
    """
    text = read_docx(file_path)
    return get_industry_scores_from_text(text, prompt)

def get_score_for_industry(scores_dict, industry_name):
    """Returnează scorul pentru o industrie (case-insensitive)."""
    for key in scores_dict:
        if key.lower() == industry_name.lower():
            return scores_dict[key]
    return 0

def get_explanation_for_industry(explanations_dict, industry_name):
    """Returnează explicația pentru o industrie (case-insensitive)."""
    for key in explanations_dict:
        if key.lower() == industry_name.lower():
            return explanations_dict[key]
    return "Nu există explicație pentru această industrie."


# docx_path = "/Users/oanarepede/Desktop/Accesa/ai_hackathon/ai_hackathon/DataSet/cv/cv_1_Florin_Neagu.docx"
# scores, explanations = parse_industry(docx_path, cv_prompt)

# print("Scoruri:", scores)
# industry_query = "Retail"
# print(f"Score pentru {industry_query}:", get_score_for_industry(scores, industry_query))

# docx_path = "/Users/oanarepede/Desktop/Accesa/ai_hackathon/ai_hackathon/DataSet/job_descriptions/job_description_2_Project Manager.docx"
# scores, explanations = parse_industry(docx_path, jd_prompt)

# print("\nScoruri:", scores)
# industry_query = "Healthcare"
# print(f"Score pentru {industry_query}:", get_score_for_industry(scores, industry_query))
