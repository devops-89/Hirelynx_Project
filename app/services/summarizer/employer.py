import re
import logging
from typing import Optional
from .base import _llm_generate, scrape_website_text

logger = logging.getLogger(__name__)

async def summarize_employer_profile(employer_data: dict) -> str:
    """Generates and enhances a personalized company summary for the employer profile using web scraping (if URL provided) or manual details."""
    website = (employer_data.get("companyWebsite") or employer_data.get("website") or employer_data.get("websiteUrl") or employer_data.get("company_website") or "").strip()
    company_name = (employer_data.get("companyName") or employer_data.get("legalName") or "").strip()
    description = (employer_data.get("companyDescription") or "").strip()
    industry, company_type, company_size = (employer_data.get("industry") or "").strip(), (employer_data.get("companyType") or "").strip(), (employer_data.get("companySize") or "").strip()
    city, province, country = (employer_data.get("city") or "").strip(), (employer_data.get("province") or "").strip(), (employer_data.get("country") or "").strip()
    location_str = ", ".join([p for p in [city, province, country] if p])

    scraped_text = await scrape_website_text(website, timeout=30.0) if website else None
    
    # Allow enhancement if there is ANY useful input: website, description, company name, industry, etc.
    has_any_info = bool(website or scraped_text or description or company_name or industry or company_type or location_str)

    if not has_any_info:
        raise ValueError("Please provide at least a Company Name, Industry, Description, or Website URL to generate an AI summary.")

    known_facts = []
    if company_name: known_facts.append(f'Company name: "{company_name}"')
    if industry: known_facts.append(f"Industry: {industry}")
    if company_type: known_facts.append(f"Company type: {company_type}")
    if company_size: known_facts.append(f"Team size: {company_size} employees")
    if location_str: known_facts.append(f"Location: {location_str}")
    if description: known_facts.append(f'Employer description / manually entered details: "{description}"')

    facts_block = "\n".join(f"  • {f}" for f in known_facts) if known_facts else "  • (no additional details provided)"
    scraped_block = (
        f"\n\nCONTENT SCRAPED FROM WEBSITE ({website}):\n{scraped_text[:2000]}\n(TRUST THIS PRIMARY SOURCE)"
        if scraped_text
        else f"\n\n(Note: Website {website} provided but scraping returned no content. Rely on and enhance the manually entered details above.)"
        if website
        else "\n\n(Note: No website URL provided. Intelligently enhance and expand the manually entered company details above into a complete, professional profile.)"
    )

    prompt = f"""You are a top-tier creative director and corporate storyteller writing for a modern recruitment platform.
Craft a 100% unique, engaging, and dynamic employer brand summary (5 to 8 sentences).

COMPANY DETAILS & INPUT DATA:
{facts_block}{scraped_block}

Requirements for Dynamic AI Generation:
1. HIGH VARIATION & CREATIVITY: Make this summary distinct, fresh, and uniquely structured. Vary the opening hook, phrasing, sentence structure, and narrative flow so every summary is completely one-of-a-kind.
2. DEEP ENHANCEMENT: Intelligently expand upon the handwritten description, industry context, and company goals into a compelling corporate narrative.
3. TONAL QUALITY: Professional, authentic, third-person perspective. Avoid repetitive boilerplate or rigid sentence templates.
4. NO CLICHÉS: Strictly avoid generic buzzwords like "dynamic", "synergy", "cutting-edge", "leveraging", "world-class", or "next-gen".
5. FACTUAL INTEGRITY: Ground the narrative in the provided details, while painting an inspiring picture of the company's culture, mission, and team.

Return ONLY the final enhanced profile summary text."""

    result = _llm_generate(prompt, max_tokens=800, temperature=0.95)
    if result:
        result = re.sub(r'\\(["\'/])', r'\1', result)
        result = re.sub(r'\\n', ' ', result)
        return " ".join(result.split())

    # Deterministic fallback — Clean factual 4-5 sentence summary (when LLM key is absent)
    sentences = []
    c_name = company_name if company_name else "The company"

    # Junk Filter for description
    is_junk = False
    if description:
        desc_clean = description.strip()
        if " " not in desc_clean or len(desc_clean) < 15 or not re.search(r'[aeiouAEIOU]', desc_clean):
            is_junk = True

    # Sentence 1 — Identity hook
    identity_parts = [f"{c_name}"]
    if company_type:
        identity_parts.append(f"is a {company_type.lower()}")
    else:
        identity_parts.append("is an organization")
    
    if industry:
        identity_parts.append(f"operating within the {industry} sector")
    
    if location_str:
        identity_parts.append(f"based in {location_str}")
    sentences.append(" ".join(identity_parts) + ".")

    # Sentence 2 — Core focus or mission
    if description and not is_junk:
        desc_snippet = description.strip().rstrip(".")
        sentences.append(f"The company is focused on {desc_snippet[:250]}.")
    elif industry:
        sentences.append(f"{c_name} provides specialized services and solutions tailored to the {industry} industry.")
    else:
        sentences.append(f"{c_name} is dedicated to delivering high-quality operational solutions across its core business areas.")

    # Sentence 3 — Team and Scale
    if company_size:
        sentences.append(f"Supported by a team of {company_size} employees, the organization emphasizes collaborative execution and reliable service delivery.")
    else:
        sentences.append(f"Their team focuses on maintaining strong operational standards and high customer satisfaction.")

    # Sentence 4 — Growth and Hiring
    sentences.append(f"{c_name} is actively seeking qualified professionals to join their team and support their ongoing business goals.")

    return " ".join(sentences)
