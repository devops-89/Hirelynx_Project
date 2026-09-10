from .candidate import DBCandidate
from .job import DBJob, DBNocOccupation
from .match import DBMatch
from .interview import DBJobInterview, DBMockInterview

# Import schemas for backward compatibility
from app.schemas.candidate import (
    CandidateProfile, PersonalDetails, Education, WorkExperience, 
    Skill, Project, Certificate, Capabilities, ResumeParseStatus
)
from app.schemas.job import (
    JobProfile, ScreeningQuestion, CandidateSearchFilters, 
    CandidateSearchRequest, NocPersonalizeRequest
)
from app.schemas.match import ResumeMatchRequest, MatchScore
from app.schemas.employer import EmployerProfile
