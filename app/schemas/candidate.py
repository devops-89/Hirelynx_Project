from datetime import datetime
from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field

class ResumeParseStatus(str, Enum):
    PENDING = "PENDING"
    PARSED = "PARSED"
    FAILED = "FAILED"

class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"
    PREFER_NOT_TO_SAY = "PREFER_NOT_TO_SAY"

class EmploymentType(str, Enum):
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACT = "CONTRACT"
    INTERNSHIP = "INTERNSHIP"
    FREELANCE = "FREELANCE"

class PersonalDetails(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    dateOfBirth: Optional[str] = None
    gender: Optional[str] = None
    openToRelocate: Optional[bool] = None

class Education(BaseModel):
    degree: str
    institution: str
    fieldOfStudy: Optional[str] = None
    country: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    currentlyStudying: Optional[bool] = None

class WorkExperience(BaseModel):
    companyName: Optional[str] = None
    role: Optional[str] = None
    jobTitle: Optional[str] = None
    location: Optional[str] = None
    employmentType: Optional[EmploymentType] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    currentlyWorking: Optional[bool] = None
    responsibilities: List[str] = []

class Skill(BaseModel):
    name: str
    level: Optional[str] = None
    yearsOfExperience: Optional[float] = None

class Project(BaseModel):
    title: str
    domain: Optional[str] = None
    tools: List[str] = []
    projectLink: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    summary: Optional[str] = None

class Certificate(BaseModel):
    name: str
    issuer: Optional[str] = None
    issueDate: Optional[str] = None
    certificateS3Key: Optional[str] = None
    certificateLink: Optional[str] = None

class Capabilities(BaseModel):
    professionalSummary: Optional[str] = Field(None, max_length=150)
    strengths: Optional[str] = Field(None, max_length=150)
    additionalDetailsS3Key: Optional[str] = None

class CandidateProfile(BaseModel):
    id: Optional[int] = None
    token: Optional[str] = None
    userId: Optional[int] = None
    personalDetails: Optional[PersonalDetails] = None
    education: Optional[List[Education]] = []
    workExperience: Optional[List[WorkExperience]] = []
    skills: Optional[List[Skill]] = []
    projects: Optional[List[Project]] = []
    certifications: Optional[List[Certificate]] = []
    capabilities: Optional[Capabilities] = None
    resumeS3Key: Optional[str] = None
    resumeParseStatus: Optional[ResumeParseStatus] = None
    resumeParsedJson: Optional[Any] = None
    resumeLastParsedAt: Optional[datetime] = None
    isProfileCompleted: bool = False
    subscriptions: Optional[Any] = []

    class Config:
        from_attributes = True
