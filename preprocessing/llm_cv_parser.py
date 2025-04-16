from openai import OpenAI
from dotenv import load_dotenv
from preprocessing.prompt_templates import CV_EXTRACTION_PROMPT
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

def parse_cv_with_llm(text: str) -> dict:
    prompt = CV_EXTRACTION_PROMPT.format(cv_text=text)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        raw = response.choices[0].message.content.strip()
        print("\n🧠 Raw CV response:\n", raw)

        response_json = json.loads(raw)

        return {
            "name": response_json.get("Full Name"),
            "skills": normalize_list(response_json.get("Technical Skills")),
            "technologies": normalize_list(response_json.get("Tools & Technologies")),
            "industries": normalize_list(response_json.get("Industries")),
            "experience_years": response_json.get("Estimated years of experience"),
            "summary": response_json.get("Short summary")
        }

    except json.JSONDecodeError as e:
        print("❌ JSON decode error in CV:", e)
        print("Raw response:", raw)
        return {}
    except Exception as e:
        print("❌ Unexpected error in CV parsing:", e)
        return {}
