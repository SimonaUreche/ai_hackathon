
from openai import OpenAI
from dotenv import load_dotenv
import os
import json 
from preprocessing.prompt_templates import JOB_EXTRACTION_PROMPT

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_job_with_llm(job_text: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "job_title": {"type": "string"},
            "required_skills": {
                "type": "array",
                "items": {"type": "string"}
            },
            "industry": {"type": "string"},
            "years_of_experience_required": {"type": "string"},
            "summary": {"type": "string"}
        },
        "required": ["job_title", "required_skills", "industry", "years_of_experience_required", "summary"],
        "additionalProperties": False
    }

    prompt = JOB_EXTRACTION_PROMPT.format(job_text=job_text)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=[{
            "type": "function",
            "function": {
                "name": "extract_job_data",
                "description": "Extracts structured data from a job description.",
                "parameters": schema,
                "strict": True
            }
        }],
        tool_choice="required"
    )

    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
