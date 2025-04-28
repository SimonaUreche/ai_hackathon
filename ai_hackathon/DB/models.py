from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class CV(Base):
    __tablename__ = 'cv'
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    # poți adăuga și alte coloane relevante

    industry_scores = relationship("CVIndustryScore", back_populates="cv")

class CVIndustryScore(Base):
    __tablename__ = 'cv_industry_score'
    id = Column(Integer, primary_key=True)
    cv_id = Column(Integer, ForeignKey('cv.id'))
    industry = Column(String)
    score = Column(Integer)
    explanation = Column(Text)

    cv = relationship("CV", back_populates="industry_scores")

class JobDescription(Base):
    __tablename__ = 'job_description'
    id = Column(Integer, primary_key=True)
    filename = Column(String)  # Numele fișierului sau un identificator
    text = Column(Text)        # Textul complet al job description-ului

    industries = relationship("JobIndustryScore", back_populates="job_description")

class JobIndustryScore(Base):
    __tablename__ = 'job_industry_score'
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('job_description.id'))
    industry = Column(String)
    score = Column(Integer)
    explanation = Column(Text)

    job_description = relationship("JobDescription", back_populates="industries")