from preprocessing.prompt_templates import CV_EXTRACTION_PROMPT
import openai
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
def parse_cv_with_llm(text: str) -> dict:
    prompt = CV_EXTRACTION_PROMPT.format(cv_text=text)

    # using OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    #JSON
    try:
        print(response['choices'][0]['message']['content'])
        response_json = json.loads(response['choices'][0]['message']['content'])
        return {
            'name': response_json.get('Full Name'),
            'skills': response_json.get('Technical Skills'),
            'technologies': response_json.get('Tools & Technologies'),
            'industries': response_json.get('Industries'),
            'experience_years': response_json.get('Estimated years of experience'),
            'summary': response_json.get('Short summary')
        }
    except Exception as e:
        print("Failed to parse CV:", e)
        return {}
