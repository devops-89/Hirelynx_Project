from typing import List, Optional
from pydantic import BaseModel

class ScreeningQuestion(BaseModel):
    id: str
    question: str
    type: str

class JobProfile(BaseModel):
    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    employmentType: Optional[str] = None
    experienceLevel: Optional[str] = None
    experienceMin: Optional[float] = None
    experienceMax: Optional[float] = None
    compensationType: Optional[str] = None
    salaryMin: Optional[float] = None
    salaryMax: Optional[float] = None
    currency: Optional[str] = None
    responsibilities: Optional[List[str]] = []
    reportingTo: Optional[str] = None
    workSchedule: Optional[str] = None
    requiredSkills: Optional[List[str]] = []
    requiresWorkAuthorization: Optional[bool] = None
    openToInternationalCandidates: Optional[bool] = None
    screeningQuestions: Optional[List[ScreeningQuestion]] = []
    isRemote: Optional[bool] = None
    expiresAt: Optional[str] = None
    
    # legacy fields just in case
    token: Optional[str] = None
    skills: Optional[List[str]] = []
    jobS3Key: Optional[str] = None

    class Config:
        from_attributes = True
        extra = "allow"

class CandidateSearchFilters(BaseModel):
    industry:      Optional[str]       = None
    companies:     Optional[List[str]] = None
    locations:     Optional[List[str]] = None
    skills:        Optional[List[str]] = None
    jobType:       Optional[List[str]] = None
    experienceMin: Optional[float]     = None
    experienceMax: Optional[float]     = None
    salaryMin:     Optional[float]     = None
    salaryMax:     Optional[float]     = None
    minMatchScore: Optional[float]     = 0.0

    class Config:
        extra = "allow"

class CandidateSearchRequest(BaseModel):
    mode:    str            = "filter"
    query:   Optional[str] = None
    filters: Optional[CandidateSearchFilters] = None
    limit:   int            = 20

    class Config:
        extra = "allow"

class NocPersonalizeRequest(BaseModel):
    nocCode: str
    jobTitle: str
    companyName: Optional[str] = None
    category: Optional[str] = None
