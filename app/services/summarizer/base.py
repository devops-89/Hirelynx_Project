import re
import random
import logging
import httpx
from typing import List, Optional, Dict, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM helper — Groq for all summarization tasks
# ---------------------------------------------------------------------------
def _llm_generate(prompt: str, max_tokens: int = 400, temperature: float = 0.95) -> Optional[str]:
    """
    Generate text via Groq API. Returns None on failure so callers can fallback.
    """
    # 1. Try Groq (Primary)
    try:
        from groq import Groq
        api_key = getattr(settings, "GROQ_API_KEY", None)
        if api_key:
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq LLM failed (Rate Limit/Down). Routing to Gemini: {e}")

    # 2. Try Gemini (Fallback)
    try:
        import google.generativeai as genai
        gemini_api_key = getattr(settings, "GEMINI_API_KEY", None)
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-3-flash-preview')
            # Use at least 1024 tokens so summaries are never cut mid-sentence
            gemini_max_tokens = max(max_tokens, 1024)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=gemini_max_tokens,
                    temperature=temperature,
                )
            )
            if response and response.text:
                logger.info("Successfully generated summary via Gemini fallback.")
                return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini Fallback also failed: {e}")

    return None

# ---------------------------------------------------------------------------
# Universal profession domain clusters
# ---------------------------------------------------------------------------
_DOMAIN_CLUSTERS: Dict[str, List[str]] = {
    "software development": ["python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust", "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring", "backend", "frontend", "full stack", "fullstack", "software engineer", "software developer", "programmer", "coding", "web development"],
    "cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins", "ci/cd", "devops", "cloud", "infrastructure", "sre", "linux", "helm", "prometheus", "grafana", "cloud engineer", "platform engineer"],
    "data & analytics": ["sql", "pandas", "spark", "hadoop", "dbt", "airflow", "bigquery", "snowflake", "tableau", "powerbi", "looker", "data analyst", "data engineer", "business analyst", "bi developer", "etl", "data warehouse"],
    "AI & machine learning": ["machine learning", "deep learning", "nlp", "pytorch", "tensorflow", "scikit-learn", "bert", "gpt", "llm", "generative ai", "mlops", "computer vision", "data scientist", "ml engineer", "ai engineer", "huggingface", "langchain", "xgboost"],
    "mobile development": ["ios", "android", "flutter", "react native", "swift", "kotlin", "xcode", "mobile developer", "app developer", "jetpack compose"],
    "cybersecurity": ["cybersecurity", "penetration testing", "soc", "siem", "owasp", "fortify", "sonarqube", "compliance", "vulnerability", "firewall", "security analyst", "infosec", "ethical hacking", "network security"],
    "product management": ["product manager", "product owner", "roadmap", "agile", "scrum", "stakeholder", "kpi", "okr", "user story", "sprint", "backlog", "go-to-market", "product strategy", "ux", "user research"],
    "healthcare & nursing": ["nurse", "nursing", "rn", "lpn", "patient care", "clinical", "icu", "er", "medication", "vital signs", "emr", "ehr", "healthcare", "hospital", "medical", "phlebotomy", "caregiver", "health aide"],
    "medicine & pharmacy": ["physician", "doctor", "md", "pharmacist", "pharmacy", "prescription", "diagnosis", "treatment", "surgery", "radiology", "lab technician", "physiotherapy", "occupational therapy", "dental", "dentist", "optometry"],
    "culinary & food service": ["chef", "cook", "culinary", "kitchen", "food preparation", "menu", "catering", "pastry", "baking", "sous chef", "line cook", "food safety", "haccp", "restaurant", "barista", "bartender", "sommelier", "banquet", "food and beverage"],
    "hospitality & tourism": ["hotel", "hospitality", "front desk", "concierge", "housekeeping", "guest services", "travel", "tourism", "resort", "event coordinator", "reservation", "property management"],
    "skilled trades": ["plumber", "plumbing", "electrician", "electrical", "carpenter", "carpentry", "welder", "welding", "hvac", "mechanic", "technician", "pipefitter", "millwright", "ironworker", "sheet metal", "mason", "tiler", "roofer", "painter", "drywaller", "gas fitter", "red seal", "apprentice", "journeyman"],
    "construction": ["construction", "site supervisor", "project manager", "general contractor", "foreman", "estimator", "blueprint", "autocad", "scheduling", "site management", "project coordination", "building", "renovation"],
    "finance & accounting": ["accountant", "accounting", "cpa", "bookkeeper", "bookkeeping", "tax", "audit", "financial analyst", "cfo", "controller", "accounts payable", "accounts receivable", "payroll", "quickbooks", "xero", "sage", "gaap", "ifrs", "financial reporting", "budget", "forecasting"],
    "banking & investment": ["banking", "bank", "investment", "portfolio", "wealth management", "equity", "fixed income", "risk management", "trading", "broker", "financial advisor", "mutual fund", "compliance", "aml", "capital markets"],
    "education & teaching": ["teacher", "teaching", "educator", "instructor", "professor", "curriculum", "lesson plan", "classroom", "student", "school", "academic", "tutor", "e-learning", "training", "workshop", "educational technology"],
    "legal": ["lawyer", "attorney", "solicitor", "barrister", "paralegal", "legal assistant", "contract", "litigation", "compliance", "corporate law", "real estate law", "immigration", "intellectual property", "dispute resolution", "arbitration"],
    "marketing & growth": ["marketing", "digital marketing", "seo", "sem", "social media", "content marketing", "email marketing", "brand", "advertising", "campaign", "google ads", "meta ads", "crm", "hubspot", "salesforce", "lead generation"],
    "sales & business dev": ["sales", "account executive", "business development", "bdr", "sdr", "closing", "pipeline", "quota", "revenue", "b2b", "b2c", "client relations", "customer success", "territory management"],
    "design & creative": ["graphic designer", "ui designer", "ux designer", "product designer", "illustrator", "figma", "sketch", "adobe", "photoshop", "indesign", "motion graphics", "video editing", "photography", "brand design", "typography"],
    "logistics & operations": ["logistics", "supply chain", "warehouse", "inventory", "forklift", "shipping", "receiving", "dispatch", "fleet management", "procurement", "vendor management", "operations manager", "lean", "six sigma"],
    "human resources": ["hr", "human resources", "recruiting", "recruiter", "talent acquisition", "onboarding", "offboarding", "hris", "workday", "bamboohr", "compensation", "benefits", "employee relations", "labour relations", "performance management"],
}

_DOMAIN_LABELS: Dict[str, str] = {k: k for k in _DOMAIN_CLUSTERS}

_EXP_LEVEL_LABELS: Dict[str, str] = {
    "ENTRY": "entry-level candidates", "JUNIOR": "junior professionals (0–2 years)", "ONE_TO_THREE": "candidates with 1–3 years of experience",
    "TWO_TO_FIVE": "mid-level professionals (2–5 years)", "THREE_TO_FIVE": "professionals with 3–5 years of experience",
    "FIVE_TO_TEN": "seasoned professionals with 5–10 years of experience", "MID": "mid-level professionals",
    "SENIOR": "senior professionals with 5+ years of experience", "LEAD": "lead-level professionals",
    "MANAGER": "managers and team leads", "DIRECTOR": "director-level professionals", "EXECUTIVE": "executive-level leaders",
    "OVER_TEN": "highly experienced professionals with 10+ years",
}

_EMPLOYMENT_LABELS: Dict[str, str] = {
    "FULL_TIME": "full-time", "PART_TIME": "part-time", "CONTRACT": "contract",
    "INTERNSHIP": "internship", "TEMPORARY": "temporary", "FREELANCE": "freelance",
}

_WORK_SCHEDULE_LABELS: Dict[str, str] = {
    "DAY": "day shift", "EVENING": "evening shift", "NIGHT": "night shift",
    "ROTATING": "rotating shifts", "FLEXIBLE": "flexible hours", "WEEKENDS": "weekends",
}

def _infer_domains(skill_names: List[str], job_title: str) -> List[str]:
    combined_terms = [s.lower() for s in skill_names] + [job_title.lower()]
    scores: Dict[str, int] = {}
    for domain, keywords in _DOMAIN_CLUSTERS.items():
        score = 0
        for kw in keywords:
            pattern = re.escape(kw) if " " in kw else rf"\b{re.escape(kw)}\b"
            for term in combined_terms:
                if re.search(pattern, term):
                    score += 1
                    break
        if score > 0:
            scores[domain] = score
    return sorted(scores, key=lambda k: scores[k], reverse=True)[:2]

def _title_to_domain_str(job_title: str) -> str:
    t = job_title.lower().strip()
    fallbacks = [
        (r"\b(chef|cook|pastry|baker|culinary)\b", "the culinary arts"),
        (r"\b(plumb|hvac|pipefitter|gas fitter)\b", "plumbing and mechanical systems"),
        (r"\b(electri|wiring|electrician)\b", "electrical systems and installation"),
        (r"\b(nurse|nursing|rn|lpn)\b", "healthcare and patient care"),
        (r"\b(doctor|physician|surgeon|md)\b", "medicine and clinical practice"),
        (r"\b(teacher|instructor|educator|tutor)\b", "education and curriculum delivery"),
        (r"\b(lawyer|attorney|paralegal|legal)\b", "legal practice and advisory"),
        (r"\b(accountant|cpa|bookkeep|auditor)\b", "accounting and financial reporting"),
        (r"\b(designer|ux|ui|creative|illustrator)\b", "design and creative direction"),
        (r"\b(sales|account exec|bdr|sdr)\b", "sales and business development"),
        (r"\b(logistics|warehouse|supply chain)\b", "logistics and supply chain management"),
        (r"\b(hr|recruiter|talent|human resource)\b", "human resources and people operations"),
        (r"\b(market|seo|brand|advertising)\b", "marketing and brand growth"),
        (r"\b(construct|foreman|site super|estimat)\b", "construction and project delivery"),
        (r"\b(carpenter|welder|mason|roofer)\b", "skilled trades and craftsmanship"),
        (r"\b(mechanic|technician|auto)\b", "automotive and mechanical services"),
    ]
    for pattern, label in fallbacks:
        if re.search(pattern, t):
            return label
    return f"the {job_title.strip().lower()} field" if job_title else "their professional field"

def _infer_years(work_experience: List[Dict]) -> Optional[int]:
    total = 0
    for exp in work_experience:
        start, end, curr = exp.get("startDate", ""), exp.get("endDate", ""), exp.get("currentlyWorking", False)
        sy = re.search(r"\b(20\d{2}|19\d{2})\b", str(start))
        ey = re.search(r"\b(20\d{2}|19\d{2})\b", str(end))
        if sy:
            s_yr = int(sy.group(1))
            e_yr = 2026 if curr else (int(ey.group(1)) if ey else s_yr + 1)
            total += max(0, e_yr - s_yr)
    return total if total > 0 else None

def _pick(pool: List[str]) -> str:
    return random.choice(pool)

async def scrape_website_text(url: str, timeout: float = 30.0) -> Optional[str]:
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        headers = {"User-Agent": "Mozilla/5.0 (compatible; HirelynxBot/1.0; +https://hirelynx.com)", "Accept": "text/html,application/xhtml+xml"}
        def _extract_text(html: str) -> str:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "img"]): tag.decompose()
            for s_id in ("about", "mission", "company", "overview", "who-we-are"):
                el = soup.find(id=s_id) or soup.find(class_=s_id)
                if el:
                    text = el.get_text(separator=" ", strip=True)
                    if len(text) > 100: return text[:2500]
            return (soup.find("body") or soup).get_text(separator=" ", strip=True)[:2500]
        extracted = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=True) as client:
            try:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    t = _extract_text(resp.text)
                    if t: extracted.append(t)
            except Exception: pass
            if len(" ".join(extracted)) < 300:
                base = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(url))
                for path in ["/about", "/about-us", "/company"]:
                    try:
                        resp = await client.get(urljoin(base, path), headers=headers)
                        if resp.status_code == 200:
                            t = _extract_text(resp.text)
                            if t: extracted.append(t); break
                    except Exception: continue
        combined = re.sub(r"\s+", " ", " ".join(extracted)).strip()
        return combined[:2500] if combined else None
    except Exception as e:
        logger.warning(f"Website scrape failed for {url}: {e}")
        return None
