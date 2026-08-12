from fastapi import APIRouter, HTTPException, Request
from app.services.summarizer import summarize_employer_profile
from app.models import EmployerProfile

router = APIRouter(prefix="/employer", tags=["Employer"])

@router.post("/company-profile")
async def create_employer_company_profile(
    request: Request,
):
    """
    Accepts either:
      - The full login response body (with a nested "employerProfile" key), OR
      - Just the employerProfile object directly

    Steps:
      1. Extracts employer/company data from whichever shape is sent.
      2. Scrapes the company website (if URL is provided).
      3. Sends manually entered details (and scraped content if available) to Groq/Gemini LLM to generate/enhance summary.
      4. Returns a personalized, AI-enhanced company summary.

    No auth required — Node.js backend calls this directly after employer login.

    Response: { "summary": "..." }
    """
    try:
        body = await request.json()

        # Handle both shapes: full login response OR just the employerProfile object
        employer_dict = body.get("employerProfile") or body.get("data", {}).get("employerProfile") or body

        # Merge top-level firstName/lastName if present (from the user object)
        if "firstName" not in employer_dict or not employer_dict.get("firstName"):
            employer_dict = {**employer_dict}
            employer_dict.setdefault("firstName", body.get("firstName") or body.get("data", {}).get("firstName"))
            employer_dict.setdefault("lastName",  body.get("lastName")  or body.get("data", {}).get("lastName"))

        # Validate with Pydantic
        profile = EmployerProfile(**employer_dict)
        summary = await summarize_employer_profile(profile.model_dump())
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
