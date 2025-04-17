
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
from preprocessing.prompt_templates import CV_EXTRACTION_PROMPT

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_cv_with_llm(cv_text: str) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "skills": {"type": "string"},
            "technologies": {
                "type": "array",
                "items": {"type": "string"}
            },
            "industries": {
                "type": "array",
                "items": {"type": "string"}
            },
            "experience_years": {"type": "string"},
            "summary": {"type": "string"}
        },
        "required": ["name", "skills", "technologies", "industries", "experience_years", "summary"],
        "additionalProperties": False
    }

    prompt = CV_EXTRACTION_PROMPT.format(cv_text=cv_text)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=[{
            "type": "function",
            "function": {
                "name": "extract_cv_data",
                "description": "Extracts structured data from a CV.",
                "parameters": schema,
                "strict": True
            }
        }],
        tool_choice="required"
    )

    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)
