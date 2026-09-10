from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from app.db.session import Base


class DBJobInterview(Base):
    """ORM model for the job_interviews table."""
    __tablename__ = "job_interviews"

    id            = Column(Integer, primary_key=True, index=True)
    transcription = Column(Text, nullable=True)   # Raw interview transcript (preferred LLM input)
    summary       = Column(Text, nullable=True)   # AI-generated summary (output — written by this API)
    summary_notes = Column(Text, nullable=True)   # Human/AI notes (fallback input if no transcript)
    createdAt     = Column(DateTime, default=datetime.utcnow)
    updatedAt     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DBMockInterview(Base):
    """ORM model for the mock-interviews table."""
    __tablename__ = "mock-interviews"

    id            = Column(Integer, primary_key=True, index=True)
    transcription = Column(Text, nullable=True)   # Raw interview transcript (preferred LLM input)
    summary       = Column(Text, nullable=True)   # AI-generated summary (output — written by this API)
    summary_notes = Column(Text, nullable=True)   # Human/AI notes (fallback input if no transcript)
    createdAt     = Column(DateTime, default=datetime.utcnow)
    updatedAt     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
