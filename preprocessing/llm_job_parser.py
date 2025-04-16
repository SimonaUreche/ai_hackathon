from openai import OpenAI
from dotenv import load_dotenv
from preprocessing.prompt_templates import JOB_EXTRACTION_PROMPT
import os
import json

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_list(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(",")]
    if isinstance(value, list):
        return value
    return []

def parse_job_with_llm(text: str) -> dict:
    prompt = JOB_EXTRACTION_PROMPT.format(job_text=text)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        raw = response.choices[0].message.content.strip()
        print("\n🧠 Raw Job response:\n", raw)

        response_json = json.loads(raw)

        return {
            "job_title": response_json.get("Job Title"),
            "required_skills": normalize_list(response_json.get("Required Technical Skills")),
            "industry": response_json.get("Relevant Industry"),
            "years_of_experience_required": response_json.get("Years of experience required"),
            "summary": response_json.get("Short summary")
        }

    except json.JSONDecodeError as e:
        print("❌ JSON decode error in Job:", e)
        print("Raw response:", raw)
        return {}
    except Exception as e:
        print("❌ Unexpected error in Job parsing:", e)
        return {}
