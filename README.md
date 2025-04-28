# DevMatch: Intelligent Talent Allocation System

## Overview
DevMatch is an AI-powered system that automatically matches developers to projects based on:
- **Technical Skills** (Python, React, AWS, etc.)
- **Industry Knowledge** (Banking, Healthcare, IT, Retail)
- **Semantic Fit** between CVs and Job Descriptions

**Goal:** Optimize talent allocation to improve project success rates and team efficiency.

## Key Features
| Feature | Description |
|---------|-------------|
|  Job-to-CV Matching | Upload a JD → Get top 5 matching CVs with scores |
|  CV-to-Job Matching | Upload a CV → Find best matching jobs |
|  Industry Matching | Prioritizes candidates with relevant industry experience |
|  Custom Skill Weights | Define skill importance (weights sum to 100%) |
|  Semantic Analysis | Combines keyword + contextual understanding |
|  AI Explanations | Generated match rationales using GPT-4o |

## Matching Algorithm
**Final Score =**  
`(Industry × 0.10) + (Technical Skills × 0.30) + (Semantic Match × 0.60)`

## Tech Stack
- **Frontend**: Streamlit
- **NLP**: spaCy, Sentence-Transformers
- **ML**: Scikit-learn (TF-IDF, similarity scoring)
- **Database**: SQLite (metadata storage)
- **AI**: OpenAI GPT-4o (explanations)
