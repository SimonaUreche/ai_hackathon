from openai import OpenAI
from dotenv import load_dotenv
import json
import os
from preprocessing.prompt_templates import JOB_EXTRACTION_PROMPT

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_job_with_llm(job_text: str) -> dict:
    prompt = JOB_EXTRACTION_PROMPT.format(job_text=job_text)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    message = response.choices[0].message.content.strip()

    try:
        return json.loads(message)
    except json.JSONDecodeError:
        if "```json" in message:
            message = message.split("```json")[1].split("```")[0]
            return json.loads(message)
        elif "```" in message:
            message = message.split("```")[1]
            return json.loads(message)
        else:
            raise ValueError("Invalid JSON output from model:\n" + message)
