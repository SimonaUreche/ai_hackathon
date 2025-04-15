import openai
from preprocessing.prompt_templates import JOB_EXTRACTION_PROMPT
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def parse_job_with_llm(text: str) -> dict:
    prompt = JOB_EXTRACTION_PROMPT.format(job_text=text)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        print(response['choices'][0]['message']['content'])
        raw_response = response['choices'][0]['message']['content'].strip()
        response_json = json.loads(raw_response)
        return {
            'job_title': response_json.get('Job Title'),
            'required_skills': response_json.get('Required Technical Skills'),
            'industry': response_json.get('Relevant Industry'),
            'years_of_experience_required': response_json.get('Years of experience required'),
            'summary': response_json.get('Short summary')
        }
    except json.JSONDecodeError as e:
        print("Failed to decode JSON from LLM:", e)
        print("Raw response:", raw_response)
        return {}
    except Exception as e:
        print("Unexpected error:", e)
        return {}

