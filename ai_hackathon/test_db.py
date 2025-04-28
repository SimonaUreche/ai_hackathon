from DB.models import CV, CVIndustryScore
from DB.session import SessionLocal

session = SessionLocal()

print("Toate scorurile pe industrii din baza de date:\n")
for score in session.query(CVIndustryScore).all():
    print(f"CV id: {score.cv_id}, Industry: {score.industry}, Score: {score.score}, Explanation: {score.explanation}")

print("CV-uri în DB:")
for cv in session.query(CV).all():
    print(f"CV id: {cv.id}, nume: {cv.nume}")

session.close()
