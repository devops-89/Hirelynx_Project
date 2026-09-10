import logging
from .base import _llm_generate

logger = logging.getLogger(__name__)


def summarize_interview_transcript(transcript: str) -> str:
    """
    Generate a rich, structured AI summary from a raw interview transcript.

    Args:
        transcript: The full raw interview transcript text.

    Returns:
        A structured, human-readable summary string with sections for
        snapshot, key topics, strengths, red flags, performance table,
        and recommendation.
    """
    content = transcript.strip()
    if not content:
        raise ValueError("Transcript text is empty.")

    prompt = f"""You are a senior talent intelligence analyst writing a formal interview debrief report for a corporate hiring committee.

Read the following interview transcript carefully and produce a structured, executive-level summary.

INTERVIEW TRANSCRIPT:
\"\"\"
{content[:7000]}
\"\"\"

---

Write the report using the exact sections below. Output clean plain text only — no markdown, no asterisks, no hashtags, no bullet symbols, no pipe tables. Use plain numbered or lettered lists where needed.

SECTION 1 — CANDIDATE SNAPSHOT
Write a tight 3 to 5 sentence paragraph capturing who the candidate is, how they performed overall, and the single most defining moment or signal from this interview. Be direct and specific.

SECTION 2 — KEY TOPICS COVERED
List the main subjects and questions discussed during the interview. Each item on a new line, starting with a dash.

SECTION 3 — STANDOUT STRENGTHS
List the candidate's strongest moments. Be specific — name the exact skill, answer, or behavior. Each item on a new line, starting with a dash.

SECTION 4 — RED FLAGS AND GAPS
List weak areas, vague answers, hesitations, or missing knowledge. Be direct but professional. Each item on a new line, starting with a dash.

SECTION 5 — PERFORMANCE RATINGS
Rate the candidate on each dimension from 1 to 5. Write each rating on its own line in this exact format:
Communication: X/5 — [one line note]
Technical Depth: X/5 — [one line note]
Problem Solving: X/5 — [one line note]
Cultural Fit and Enthusiasm: X/5 — [one line note]
Confidence: X/5 — [one line note]

SECTION 6 — HIRING RECOMMENDATION
State clearly whether to advance, request a follow-up round, or pass. Write 2 to 4 sentences with a confident, evidence-backed recommendation.

---

Tone and style rules:
- Write in third-person professional corporate language ("The candidate demonstrated...", "Sharma exhibited...").
- Every claim must be grounded in something actually said or observed in the transcript.
- No filler phrases. No generic commentary. Every sentence must add value.
- The report should be useful to a hiring manager who was not present in the interview.
- Output plain text only. No markdown formatting whatsoever.
"""

    result = _llm_generate(prompt, max_tokens=1200, temperature=0.4)
    if result:
        return result.strip()

    # --- Deterministic fallback (LLM unavailable) ---
    logger.warning("LLM unavailable — returning deterministic fallback summary.")
    word_count = len(content.split())
    excerpt = content[:350].strip()
    return (
        f"## Interview Summary\n\n"
        f"**Source**: Full Interview Transcript ({word_count} words captured)\n\n"
        f"**Note**: AI summary generation is temporarily unavailable. "
        f"Please review the transcript manually.\n\n"
        f"**Excerpt**:\n> {excerpt}..."
    )
