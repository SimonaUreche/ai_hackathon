from sqlalchemy import Column, Integer, String, Text
from .session import Base

class CV(Base):
    __tablename__ = "cvs"
    id            = Column(Integer, primary_key=True, index=True)
    filename      = Column(String, unique=True, index=True)
    skills_json   = Column(Text)
    industry_json = Column(Text)
