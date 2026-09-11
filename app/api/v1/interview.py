from fastapi import APIRouter, HTTPException, Depends, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.utils import extract_role_from_token
from app.services.summarizer.interview import summarize_interview_transcript

router = APIRouter(prefix="/interview", tags=["Interview"])
security = HTTPBearer()


@router.post("/summarize")
async def summarize_interview(
    transcript: str = Form(..., description="Paste the full interview transcript here"),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Generate an AI-powered summary from a raw interview transcript.

    - **Auth**: Requires a valid ADMIN Bearer token.
    - **Body**: `{ "transcript": "<raw interview transcript text>" }`
    - **Returns**: `{ "summary": "<structured AI summary>" }`

    The summary includes:
      - Snapshot of the interview
      - Key topics covered
      - Standout strengths
      - Red flags & gaps
      - Performance breakdown table
      - Recommendation
    """
    # --- Auth: ADMIN only ---
    token = credentials.credentials
    role = extract_role_from_token(token)
    if role != "ADMIN":
        raise HTTPException(status_code=403, detail="Access denied. ADMIN role required.")

    # --- Validate transcript ---
    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty.")


    # --- Generate summary ---
    try:
        summary = summarize_interview_transcript(transcript=transcript)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

    return {"summary": summary}

