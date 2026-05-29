import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Data Helpers ---

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()

# --- Semantic Scoring Engine ---

class ScoringEngine:
    """
    Professional Multi-Factor Scoring Engine.
    Uses a hybrid approach:
    1. Semantic Similarity (Sentence-BERT) for Responsibilities.
    2. N-Gram & Substring Matching for Skills.
    3. Logic-based Experience & Education validation.
    """

    # Weights (Strictly following original requirements)
    WEIGHT_SKILLS  = 0.50
    WEIGHT_EXP     = 0.15
    WEIGHT_EDU     = 0.05
    WEIGHT_RESP    = 0.30

    def __init__(self, weight: float = 0.5):
        self.weight = weight
        try:
            # Loaded once at startup
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            self.encoder = None

    def calculate_score(
        self,
        parsed_json: Dict[str, Any],
        resume_text: str,
        job_description: str,
        required_skills: List[str],
        job_responsibilities: List[str],
        experience_min: Optional[float] = 0,
        experience_max: Optional[float] = None,
    ) -> Dict[str, Any]:
        """The primary entry point for scoring a candidate."""
        
        # 1. Extract Candidate Facts
        c_years = self._extract_years(parsed_json, resume_text)
        c_skills = self._extract_skills(parsed_json, resume_text)
        c_edu_tier = self._extract_edu_tier(parsed_json)
        c_resps = self._extract_responsibilities(parsed_json, resume_text)

        # 2. Factor 1: Skills Match (50%)
        skills_score, matched, missing = self._score_skills(required_skills, c_skills, parsed_json, resume_text)

        # 3. Factor 2: Experience Match (15%)
        exp_score = self._score_experience(c_years, experience_min, experience_max)

        # 4. Factor 3: Education Match (5%)
        # Defaulting to 100 if no specific tier is required by the JD
        edu_score = self._score_education(c_edu_tier, job_description)

        # 5. Factor 4: Responsibilities Semantic Match (30%)
        resp_score = self._score_responsibilities(job_responsibilities, c_resps)

        # 6. Final Calculation
        final_score = float(round(
            (skills_score * self.WEIGHT_SKILLS) +
            (exp_score    * self.WEIGHT_EXP)    +
            (edu_score    * self.WEIGHT_EDU)    +
            (resp_score   * self.WEIGHT_RESP),
            2
        ))

        return {
            "score": final_score,
            "jobMatchScore": final_score,
            "matchedSkillsList": matched,
            "missingSkills": missing,
            "totalRequired": len(required_skills),
            "recommendation": self._get_recommendation(final_score),
            "breakdown": {
                "skillsScore": skills_score,
                "experienceScore": exp_score,
                "educationScore": edu_score,
                "responsibilitiesScore": resp_score,
                "candidateYears": c_years,
                "requiredYearsMin": experience_min or 0,
                "candidateRespCount": len(c_resps),
                "jobResponsibilitiesCount": len(job_responsibilities),
                "scoredBy": "hybrid-semantic-engine"
            }
        }

    # --- Scoring Components ---

    def _score_skills(self, required: List[str], candidate: List[str], data: Dict, full_text: str = "") -> Tuple[float, List[str], List[str]]:
        """Weigh skills by recency and depth, with raw text fallback."""
        if not required: return 100.0, [], []
        
        full_text_low = full_text.lower()
        
        # Build a priority map from work history
        priority_skills = set()
        for exp in data.get("workExperience", [])[:2]: # Top 2 recent roles
            role_text = f"{exp.get('role', '')} {exp.get('responsibilities', [])}".lower()
            for s in candidate:
                if s.lower() in role_text: priority_skills.add(s)

        matched = []
        missing = []
        raw_score = 0.0

        for req in required:
            req_low = req.lower().strip()
            # Direct match, or substring match, OR raw text match
            if any(req_low in c.lower() for c in candidate) or req_low in full_text_low:
                matched.append(req)
                # Boost if found in recent roles
                raw_score += 1.2 if any(req_low in p.lower() for p in priority_skills) else 1.0
            else:
                missing.append(req)
        
        score = (raw_score / len(required)) * 100
        return float(round(max(0.0, min(100.0, score)), 2)), matched, missing

    def _score_experience(self, years: float, req_min: Optional[float], req_max: Optional[float]) -> float:
        if not req_min or req_min == 0:
            return 100.0
        if years < req_min:
            return float(round((years / req_min) * 100, 2))
        return 100.0

    def _score_education(self, tier: int, jd: str) -> float:
        req_tier = 0
        jd_low = jd.lower()
        if any(k in jd_low for k in ["phd", "doctorate"]): req_tier = 6
        elif "master" in jd_low: req_tier = 5
        elif any(k in jd_low for k in ["bachelor", "degree"]): req_tier = 4
        if req_tier == 0: return 100.0
        if tier >= req_tier: return 100.0
        if tier == 0: return 50.0
        return float(round((tier / req_tier) * 100, 2))

    def _score_responsibilities(self, job_resps: List[str], cand_resps: List[str]) -> float:
        """Semantic match with a bias toward 'Impact' sentences."""
        if not job_resps or not self.encoder: return 100.0 if not job_resps else 0.0
        if not cand_resps: return 0.0
        
        try:
            # Action verbs and metrics detection
            impact_verbs = {"developed", "led", "managed", "increased", "reduced", "saved", "implemented", "scaled"}
            
            job_embs = self.encoder.encode(job_resps, convert_to_numpy=True)
            cand_embs = self.encoder.encode(cand_resps, convert_to_numpy=True)
            
            sim_matrix = cosine_similarity(job_embs, cand_embs)
            best_matches = sim_matrix.max(axis=1)
            
            # Apply impact multipliers
            weighted_matches = []
            for i, sim in enumerate(best_matches):
                # Find which candidate resp matched this job resp best
                best_idx = sim_matrix[i].argmax()
                best_cand_resp = cand_resps[best_idx].lower()
                
                multiplier = 1.0
                if any(v in best_cand_resp for v in impact_verbs): multiplier += 0.1
                if re.search(r"\d+%", best_cand_resp) or re.search(r"\$\d+", best_cand_resp): multiplier += 0.1
                
                weighted_matches.append(sim * multiplier)

            avg_sim = np.mean(weighted_matches)
            score = (float(avg_sim) - 0.15) / 0.65 * 100
            return float(round(max(0.0, min(100.0, score)), 2))
        except Exception as e:
            logger.error(f"Semantic match error: {e}")
            return 0.0

    # --- Extraction Helpers ---

    def _extract_years(self, data: Dict, raw_text: str = "") -> float:
        total = 0.0
        work_exp = data.get("workExperience", [])
        
        if work_exp:
            for exp in work_exp:
                start = str(exp.get("startDate", ""))
                end = str(exp.get("endDate", ""))
                curr = bool(exp.get("currentlyWorking", False))
                sy = re.findall(r"\d{4}", start)
                ey = re.findall(r"\d{4}", end)
                if sy:
                    s_yr = int(sy[0])
                    e_yr = 2026 if curr else (int(ey[0]) if ey else s_yr + 1)
                    total += max(0.5, float(e_yr - s_yr))
        
        # Fallback: Scan raw text for phrases like "5 years", "2018-2022", etc.
        if total == 0 and raw_text:
            years_found = re.findall(r"(\d{1,2})\+?\s*years?", raw_text.lower())
            if years_found:
                total = max(float(y) for y in years_found)
            else:
                # Last resort: Count unique years mentioned
                all_years = re.findall(r"\b(20\d{2}|19\d{2})\b", raw_text)
                if all_years:
                    unique_years = sorted(list(set(int(y) for y in all_years)))
                    total = max(1.0, float(len(unique_years)))

        return round(total, 1)

    def _extract_skills(self, data: Dict, text: str) -> List[str]:
        # Merge skills list and project tools
        skills = [str(s) for s in data.get("skills", []) if s]
        for proj in data.get("projects", []):
            skills.extend([str(t) for t in proj.get("tools", []) if t])
        
        # If skills list is empty, return a very basic split of text for safety
        if not skills and text:
            # Simple heuristic: capitalized words might be skills
            potential = re.findall(r"\b[A-Z][a-zA-Z0-9+#.]+\b", text)
            return list(set(potential))
            
        return list(set(skills))

    def _extract_edu_tier(self, data: Dict) -> int:
        tiers = []
        for edu in data.get("education", []):
            deg = str(edu.get("degree", "")).lower()
            if any(k in deg for k in ["phd", "doctor"]): tiers.append(6)
            elif "master" in deg: tiers.append(5)
            elif "bachelor" in deg: tiers.append(4)
            elif "diploma" in deg: tiers.append(2)
        return max(tiers, default=0)

    def _extract_responsibilities(self, data: Dict, raw_text: str = "") -> List[str]:
        resps = []
        for exp in data.get("workExperience", []):
            resps.extend([str(r).strip() for r in exp.get("responsibilities", []) if r])
        
        # Fallback: If no responsibilities found in workExperience, split raw text by common bullet symbols
        if not resps and raw_text:
            # Look for lines starting with bullets or common resume action verbs
            potential = re.findall(r"(?:^|[\n\r])\s*[•\-\*]\s*(.{10,200})", raw_text)
            if potential:
                resps = potential
            else:
                # Last resort: Split by newline and keep lines that look like sentences
                lines = raw_text.split('\n')
                resps = [line.strip() for line in lines if 20 < len(line.strip()) < 300]

        # Deduplicate and clean
        return list(set(str(r).strip() for r in resps if len(str(r)) > 5))

    def _get_recommendation(self, score: float) -> str:
        if score >= 85: return "Excellent Match"
        if score >= 70: return "Strong Match"
        if score >= 55: return "Good Match"
        if score >= 40: return "Potential Match"
        return "Low Match"

    # --- Backward Compatibility ---

    def score(self, resume_text: str, jd_description: str, keywords: List[str]) -> Dict[str, Any]:
        return self.score_with_embedding(resume_text, jd_description, None, keywords, [])

    def score_with_embedding(
        self,
        resume_text: str,
        jd_description: str,
        query_embedding: Any,
        keywords: List[str],
        candidate_skills: List[str] = [],
    ) -> Dict[str, Any]:
        """Fine-tuned AI scoring for Global Candidate Search."""
        resume_text_lower = resume_text.lower()
        cand_skills_lower = [s.lower() for s in candidate_skills]

        matched = []
        raw_kw_score = 0.0
        
        for k in keywords:
            k_low = k.lower()
            if k_low in resume_text_lower:
                matched.append(k)
                # Boost if explicitly in the candidate's skills list
                if any(k_low in cs for cs in cand_skills_lower):
                    raw_kw_score += 1.5
                else:
                    raw_kw_score += 1.0

        if keywords:
            # Normalize kw_score to max 100
            kw_score = min(100.0, (raw_kw_score / len(keywords)) * 100)
        else:
            kw_score = 0.0

        sim = 0.0
        normalized_sim = 0.0
        if self.encoder and query_embedding is not None:
             try:
                 resume_emb = self.encoder.encode([resume_text], convert_to_numpy=True)
                 sim = float(cosine_similarity(query_embedding.reshape(1, -1), resume_emb.reshape(1, -1))[0][0])
                 # Normalize similarity: shift by 0.15, scale by 0.55
                 normalized_sim = max(0.0, min(100.0, (sim - 0.15) / 0.55 * 100))
             except Exception: pass
             
        # 40% Keyword Match, 60% Semantic Match
        final = round((kw_score * 0.4) + (normalized_sim * 0.6), 2)
        return {
            "score": final,
            "matched_skills": matched,
            "recommendation": self._get_recommendation(final)
        }

    # --- AI Scoring (Groq) ---

    def _groq_score(
        self,
        resume_text: str,
        job_description: str,
        required_skills: List[str],
        exp_min: Optional[float],
        exp_max: Optional[float],
        job_responsibilities: List[str],
        job_title: str,
        candidate_responsibilities: List[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            from groq import Groq
            import json as _json
            api_key = getattr(settings, "GROQ_API_KEY", None)
            if not api_key: return None

            prompt = f"""Score this candidate for the role of {job_title}. 
            JD: {job_description[:3000]}
            REQ SKILLS: {required_skills}
            REQ EXP: {exp_min} years
            
            CANDIDATE RESUME: {resume_text[:6000]}
            EXTRACTED RESP: {candidate_responsibilities[:20]}

            Return JSON:
            {{
                "skillsScore": <0-100>,
                "experienceScore": <0-100>,
                "educationScore": <0-100>,
                "responsibilitiesScore": <0-100>,
                "reasoning": "Brief explanation",
                "matchedSkills": [],
                "missingSkills": []
            }}"""

            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = _json.loads(resp.choices[0].message.content)
            return data
        except Exception as e:
            logger.error(f"Groq scoring failed: {e}")
            return None

    def score_resume_against_job_ai(self, **kwargs) -> Dict[str, Any]:
        """High-accuracy entry point using Groq + Semantic Logic."""
        parsed_json = kwargs.get("parsed_json", {})
        resume_text = kwargs.get("resume_text", "")
        job_description = kwargs.get("job_description", "")
        required_skills = kwargs.get("required_skills", [])
        job_responsibilities = kwargs.get("job_responsibilities", [])
        exp_min = kwargs.get("exp_min", 0)
        
        # Extract objective facts first
        c_years = self._extract_years(parsed_json, resume_text)
        c_resps = self._extract_responsibilities(parsed_json, resume_text)

        # Try Groq for intelligent matching
        ai_data = self._groq_score(
            resume_text=resume_text,
            job_description=job_description,
            required_skills=required_skills,
            exp_min=exp_min,
            exp_max=kwargs.get("exp_max"),
            job_responsibilities=job_responsibilities,
            job_title=kwargs.get("job_title", "Position"),
            candidate_responsibilities=c_resps
        )

        if ai_data:
            s_score = float(ai_data.get("skillsScore", 0))
            e_score = float(ai_data.get("experienceScore", 0))
            edu_score = float(ai_data.get("educationScore", 0))
            r_score = float(ai_data.get("responsibilitiesScore", 0))
            
            # Recompute weighted score for total accuracy
            final = float(round(
                (s_score * self.WEIGHT_SKILLS) +
                (r_score * self.WEIGHT_RESP) +
                (e_score * self.WEIGHT_EXP) +
                (edu_score * self.WEIGHT_EDU), 2
            ))
            
            return {
                "score": final,
                "jobMatchScore": final,
                "matchedSkillsList": ai_data.get("matchedSkills", []),
                "missingSkills": ai_data.get("missingSkills", []),
                "totalRequired": len(required_skills),
                "recommendation": self._get_recommendation(final),
                "breakdown": {
                    "skillsScore": s_score,
                    "experienceScore": e_score,
                    "educationScore": edu_score,
                    "responsibilitiesScore": r_score,
                    "overallReasoning": ai_data.get("reasoning", ""),
                    "candidateYears": c_years,
                    "requiredYearsMin": exp_min,
                    "candidateRespCount": len(c_resps),
                    "scoredBy": "groq-llm"
                }
            }

        # 3. Fallback to our Semantic Engine if Groq is down
        return self.calculate_score(
            parsed_json=parsed_json,
            resume_text=resume_text,
            job_description=job_description,
            required_skills=required_skills,
            job_responsibilities=job_responsibilities,
            experience_min=exp_min,
            experience_max=kwargs.get("exp_max")
        )
