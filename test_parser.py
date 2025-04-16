from preprocessing.file_converter import extract_text
from preprocessing.llm_cv_parser import parse_cv_with_llm
from preprocessing.llm_job_parser import parse_job_with_llm
import pprint
import json
import os

output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)

#path to your sample files
cv_path = "data/cv/cv_test.docx"
job_path = "data/job_descriptions/job_test.docx"

#extract text from both files
cv_text = extract_text(cv_path)
job_text = extract_text(job_path)

#send to LLM for structured parsing
print("\nExtracting CV data via LLM...")
cv_result = parse_cv_with_llm(cv_text)

print("\n Extracting Job Description via LLM...")
job_result = parse_job_with_llm(job_text)

pp = pprint.PrettyPrinter(indent=2, width=100)
print("\nParsed CV:")
pp.pprint(cv_result)

print("\n Parsed Job:")
pp.pprint(job_result)

cv_output_path = os.path.join(output_dir, "parsed_cv.json")
job_output_path = os.path.join(output_dir, "parsed_job.json")

with open(cv_output_path, 'w', encoding='utf-8') as f:
    json.dump(cv_result, f, indent=2, ensure_ascii=False)

with open(job_output_path, 'w', encoding='utf-8') as f:
    json.dump(job_result, f, indent=2, ensure_ascii=False)

print(f"CV: {cv_output_path}")
print(f"Job: {job_output_path}")
