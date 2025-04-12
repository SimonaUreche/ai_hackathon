# AI Hackathon - 

CV & Job Description Similarity Matcher

**12.04.2025**

## Project Description 

This project is designed to compute the similarity between a list of CVs and a list of job descriptions, both stored as separate documents. The purpose is to match candidates to the most suitable job descriptions using NLP techniques.

## Steps 

1. **Read Documents**  
   Load all CVs and job descriptions from their respective folders.

2. **Preprocess Text**  
   Clean and standardize text:
   - Lowercase conversion  
   - Removal of stopwords  
   - Punctuation removal  
   - Tokenization and lemmatization

3. **Text Vectorization**  
   Use TFIDF to convert text into numerical vectors

4. **Compute Similarity**  
   Calculate cosine similarity between all CVs and job descriptions.

5. **Display Results**  
   Output a similarity matrix that shows how well each CV matches each job description.

##Output

|        | JD_1 | JD_2 | JD_3 |
|--------|------|------|------|
| CV_1   | 0.87 | 0.35 | 0.52 |
| CV_2   | 0.22 | 0.91 | 0.49 |
| CV_3   | 0.67 | 0.43 | 0.89 |


