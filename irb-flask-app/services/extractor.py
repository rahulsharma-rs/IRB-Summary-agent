import calendar
import json
import re
import asyncio
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from config import Config
from services.embedder import embed_texts, embed_single

client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Field-specific extraction prompts - IMPROVED VERSION
FIELD_PROMPTS = {
    "irb_number": {
        "system": "You are an expert at identifying IRB protocol numbers. Extract ONLY the official IRB protocol identifier.",
        "user_template": """Question: What is the IRB protocol number in this document?

Look for the official IRB identifier (e.g., IRB-YYYYNNNNN, IRB YYYYNNNNN, IRB#NNNNN, Protocol #NNNNN).

Document excerpt:
{text}

Rules:
1) Return the exact protocol number (keep dashes/spaces), the one labeled IRB/IRB protocol.
2) Do not include version or amendment numbers.
3) If none, return null.

Return ONLY valid JSON:
{{"irb_number": "IRB-300006637"}} or {{"irb_number": null}}""",
        "max_tokens": 150,
        "search_keywords": ["irb", "protocol", "eportfolio", "protocol number", "irb number", "irb#"]
    },

    "principal_investigator": {
        "system": "You are an expert at identifying Principal Investigators. Extract the PI's full name exactly as written.",
        "user_template": """Question: Who is the Principal Investigator (PI) for this study?

Look for labels like "Principal Investigator", "PI", "Lead Investigator".

Document excerpt:
{text}

Rules:
1) Return the full name with titles/credentials if present.
2) If multiple investigators, choose the principal/primary one.
3) No contact info.
4) If none, return null.

Return ONLY valid JSON:
{{"principal_investigator": "Dr. Jane Smith, MD"}} or {{"principal_investigator": null}}""",
        "max_tokens": 200,
        "search_keywords": ["principal investigator", "pi:", "lead investigator", "investigator:", "study director"]
    },

    "study_title": {
        "system": "You are an expert at identifying official study titles. Extract the complete, formal study title.",
        "user_template": """Question: What is the official study/protocol title?

Document excerpt:
{text}

Rules:
1) Return the full formal title (include subtitles/phases).
2) Do not summarize or truncate; strip surrounding quotes.
3) If none, return null.

Return ONLY valid JSON:
{{"study_title": "Complete Study Title Here"}} or {{"study_title": null}}""",
        "max_tokens": 300,
        "search_keywords": ["study title", "project title", "protocol title", "title:", "research title"]
    },

    "study_purpose": {
        "system": "You are an expert at summarizing research purposes. Create a clear 1-2 sentence summary.",
        "user_template": """Question: What is the purpose/objective of the study? Answer in 1-2 sentences.

Document excerpt:
{text}

Rules:
1) Summarize in your own words (no verbatim copying).
2) Max 2 sentences; state what is studied and why.
3) If not found, return null.

Return ONLY valid JSON:
{{"study_purpose": "Brief 1-2 sentence summary here"}} or {{"study_purpose": null}}""",
        "max_tokens": 400,
        "search_keywords": ["purpose", "objective", "aims", "background", "research question", "rationale",
                            "hypothesis"]
    },

    "inclusion_criteria": {
        "system": "You are an expert at extracting inclusion criteria. List the key eligibility requirements.",
        "user_template": """Question: What are the inclusion criteria (who CAN participate)?

Document excerpt:
{text}

Rules:
1) Return 3-5 key bullets, concise.
2) Format with bullets using the character • and newline between items.
3) If none, return null.

Return ONLY valid JSON:
{{"inclusion_criteria": "• Criterion 1\\n• Criterion 2\\n• Criterion 3"}} or {{"inclusion_criteria": null}}""",
        "max_tokens": 500,
        "search_keywords": ["inclusion criteria", "eligibility", "eligible", "subject selection",
                            "participant criteria"]
    },

    "exclusion_criteria": {
        "system": "You are an expert at extracting exclusion criteria. List the key disqualifying factors.",
        "user_template": """Question: What are the exclusion criteria (who CANNOT participate)?

Document excerpt:
{text}

Rules:
1) Return 3-5 key bullets, concise.
2) Format with bullets using the character • and newline between items.
3) If none, return null.

Return ONLY valid JSON:
{{"exclusion_criteria": "• Criterion 1\\n• Criterion 2\\n• Criterion 3"}} or {{"exclusion_criteria": null}}""",
        "max_tokens": 500,
        "search_keywords": ["exclusion criteria", "ineligibility", "not eligible", "cannot participate", "exclusions"]
    },

    "data_elements": {
        "system": "You are an expert at identifying approved data elements and PHI identifiers.",
        "user_template": """Question: What data elements/identifiers/PHI are approved for collection or use?

Document excerpt:
{text}

Rules:
1) List the specific elements and any date ranges.
2) Include identifiers if mentioned (names, MRNs, dates, etc.).
3) Be concise.
4) If none, return null.

Return ONLY valid JSON:
{{"data_elements": "Demographics, diagnoses, lab results, treatment dates (2020-2023), patient names and MRNs"}} or {{"data_elements": null}}""",
        "max_tokens": 400,
        "search_keywords": ["data elements", "identifiers", "phi", "data requested", "date range", "data collection",
                            "variables"]
    },

    "funding_source": {
        "system": "You are an expert at identifying funding sources and sponsors.",
        "user_template": """Question: What is the funding source or sponsor?

Document excerpt:
{text}

Rules:
1) Name the organization/entity; include grant numbers if present.
2) If unfunded/none, say "Unfunded".
3) If none found, return null.

Return ONLY valid JSON:
{{"funding_source": "National Institutes of Health (NIH Grant R01-123456)"}} or {{"funding_source": null}}""",
        "max_tokens": 300,
        "search_keywords": ["funding", "sponsor", "grant", "support", "financial", "funded by", "nih", "contract"]
    },

    "protocol_status": {
        "system": "You are an expert at identifying protocol approval status. Use ONLY the allowed status values.",
        "user_template": """Question: What is the current protocol approval status?

Allowed values (lowercase): approved, pending, expired, exempt, withdrawn.

Document excerpt:
{text}

Rules:
1) Return exactly one allowed value.
2) If none found, return null.

Return ONLY valid JSON:
{{"protocol_status": "approved"}} or {{"protocol_status": null}}""",
        "max_tokens": 150,
        "search_keywords": ["status", "approval", "exempt", "expired", "withdrawn", "pending", "determination"]
    },

    "expiration_date": {
        "system": "You are an expert at finding protocol expiration or approval dates. Return dates in YYYY-MM-DD format.",
        "user_template": """Question: What is the protocol expiration date or approval end date?

Document excerpt:
{text}

Rules:
1) Return in YYYY-MM-DD; if only month/year, use last day of that month.
2) If a range is shown, return the END date.
3) If none, return null.

Return ONLY valid JSON:
{{"expiration_date": "2025-12-31"}} or {{"expiration_date": null}}""",
        "max_tokens": 150,
        "search_keywords": ["expiration", "expires", "approval date", "valid until", "continuing review",
                            "approval period"]
    }
}


def find_relevant_sections(full_text: str,
                           keywords: List[str],
                           context_chars: int = 3000,
                           pages_text: Optional[List[Tuple[int, str]]] = None,
                           field_name: Optional[str] = None) -> str:
    """
    Build context for extraction.
    Prefer semantic retrieval over pages (top 50% most similar),
    fall back to keyword-density window on the full text.
    """
    # Try semantic retrieval over pages if available
    if pages_text:
        try:
            page_texts = [txt for _, txt in pages_text if txt and txt.strip()]
            if page_texts:
                page_embeddings = embed_texts(page_texts)
                query_text = (field_name or '') + " " + " ".join(keywords)
                query_embedding = embed_single(query_text.strip() or "irb metadata")

                scores = []
                for idx, emb in enumerate(page_embeddings):
                    scores.append((idx, float(emb @ query_embedding)))

                if scores:
                    # Take top 50% of pages by similarity (at least 1)
                    scores.sort(key=lambda x: -x[1])
                    top_k = max(1, len(scores) // 2)
                    top_indices = [idx for idx, _ in scores[:top_k]]
                    selected = "\n\n".join(page_texts[i] for i in top_indices)
                    if selected:
                        return selected[:context_chars]
        except Exception:
            # If embedding lookup fails, fall back to keyword method
            pass

    # Fallback: keyword density search on full text
    text_lower = full_text.lower()

    positions = []
    for keyword in keywords:
        idx = 0
        while True:
            pos = text_lower.find(keyword.lower(), idx)
            if pos == -1:
                break
            positions.append(pos)
            idx = pos + 1

    if not positions:
        return full_text[:context_chars]

    positions.sort()
    best_start = 0
    best_count = 0

    for start_pos in positions:
        end_pos = start_pos + context_chars
        count = sum(1 for p in positions if start_pos <= p < end_pos)
        if count > best_count:
            best_count = count
            best_start = max(0, start_pos - 500)

    return full_text[best_start:best_start + context_chars]


def find_page_references(extracted_value: str, pages_text: List[Tuple[int, str]],
                         keywords: List[str]) -> List[int]:
    """
    Find which pages contain the extracted value or related keywords.

    Args:
        extracted_value: The value that was extracted
        pages_text: List of (page_number, text) tuples
        keywords: Keywords related to this field

    Returns:
        List of page numbers where this information was found
    """
    if not extracted_value or not pages_text:
        return []

    page_refs = set()
    value_lower = str(extracted_value).lower()

    # Split extracted value into meaningful tokens (remove common words)
    value_tokens = set()
    for token in re.findall(r'\b\w+\b', value_lower):
        if len(token) > 3 and token not in {'this', 'that', 'with', 'from', 'have', 'been', 'will'}:
            value_tokens.add(token)

    for page_num, page_text in pages_text:
        page_lower = page_text.lower()

        # Check if page contains the exact value (or significant part of it)
        if len(value_lower) > 10:
            # For longer values, check if 60% of it appears
            value_parts = value_lower.split()
            matches = sum(1 for part in value_parts if len(part) > 3 and part in page_lower)
            if matches >= len(value_parts) * 0.6:
                page_refs.add(page_num)
                continue

        # Check if page contains significant tokens from the value
        token_matches = sum(1 for token in value_tokens if token in page_lower)
        if token_matches >= min(3, len(value_tokens) * 0.5):
            page_refs.add(page_num)
            continue

        # Check if page contains the field-related keywords
        keyword_matches = sum(1 for kw in keywords if kw.lower() in page_lower)
        if keyword_matches >= 2:
            page_refs.add(page_num)

    return sorted(list(page_refs))


def extract_single_field(field_name: str, document_text: str, model: str = None,
                         pages_text: List[Tuple[int, str]] = None) -> Dict:
    """
    Extract a single field from the document using a focused prompt.

    Args:
        field_name: Name of the field to extract
        document_text: Full document text
        model: Model to use for extraction
        pages_text: List of (page_number, text) tuples for reference tracking

    Returns:
        dict with 'field_name', 'value', 'status', 'error', 'page_references'
    """
    if not model:
        model = Config.OPENAI_MODEL

    if field_name not in FIELD_PROMPTS:
        return {
            'field_name': field_name,
            'value': None,
            'page_references': [],
            'status': 'failed',
            'error': f'Unknown field: {field_name}'
        }

    prompt_config = FIELD_PROMPTS[field_name]

    # Find relevant sections in document
    relevant_text = find_relevant_sections(
        document_text,
        prompt_config['search_keywords'],
        pages_text=pages_text,
        field_name=field_name
    )

    user_prompt = prompt_config['user_template'].format(text=relevant_text)

    try:
        # Build API call parameters
        api_params = {
            'model': model,
            'messages': [
                {"role": "system", "content": prompt_config['system']},
                {"role": "user", "content": user_prompt}
            ],
            'response_format': {"type": "json_object"}
        }

        # Handle different parameter names based on model
        model_lower = model.lower()

        # GPT-5 and reasoning models have restrictions
        is_restricted_model = any(x in model_lower for x in ['gpt-5', 'o1', 'o3'])

        if is_restricted_model:
            # New models: use max_completion_tokens, no temperature
            api_params['max_completion_tokens'] = prompt_config['max_tokens']
        else:
            # Older models: use max_tokens and temperature
            api_params['max_tokens'] = prompt_config['max_tokens']
            api_params['temperature'] = 0.1

        response = client.chat.completions.create(**api_params)

        raw_response = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if raw_response.startswith('```'):
            raw_response = re.sub(r'^```[a-zA-Z0-9_-]*\n', '', raw_response)
            raw_response = raw_response.rstrip('`').strip()

        parsed = json.loads(raw_response)
        value = parsed.get(field_name)

        # Clean up the value
        if value and isinstance(value, str):
            value = value.strip()
            if value.lower() in ['not specified', 'n/a', 'none', 'null', 'not found', '']:
                value = None

        # Find page references if value was found
        page_refs = []
        if value and pages_text:
            page_refs = find_page_references(value, pages_text, prompt_config['search_keywords'])

        return {
            'field_name': field_name,
            'value': value,
            'page_references': page_refs,
            'status': 'success',
            'error': None
        }

    except json.JSONDecodeError as e:
        return {
            'field_name': field_name,
            'value': None,
            'page_references': [],
            'status': 'failed',
            'error': f'JSON parse error: {str(e)}'
        }
    except Exception as e:
        return {
            'field_name': field_name,
            'value': None,
            'page_references': [],
            'status': 'failed',
            'error': str(e)
        }


def extract_metadata_parallel(document_text: str, model: str = None, max_workers: int = 5,
                              pages_text: List[Tuple[int, str]] = None) -> dict:
    """
    Extract all metadata fields in parallel using ThreadPoolExecutor.

    This is the NEW recommended approach - extracts each field independently
    with focused prompts, running multiple extractions concurrently.

    Args:
        document_text: Full document text
        model: Model to use for extraction
        max_workers: Number of concurrent API calls
        pages_text: List of (page_number, text) tuples for reference tracking

    Returns:
        dict with all field values, page references, and an extraction_summary
    """
    if not model:
        model = Config.OPENAI_MODEL

    fields_to_extract = list(FIELD_PROMPTS.keys())
    results = {}
    page_references = {}
    extraction_summary = {
        'total_fields': len(fields_to_extract),
        'successful': 0,
        'failed': 0,
        'errors': []
    }

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all extraction tasks
        future_to_field = {
            executor.submit(extract_single_field, field, document_text, model, pages_text): field
            for field in fields_to_extract
        }

        # Collect results as they complete
        for future in as_completed(future_to_field):
            field = future_to_field[future]
            try:
                result = future.result()
                results[result['field_name']] = result['value']
                page_references[result['field_name']] = result.get('page_references', [])

                if result['status'] == 'success':
                    extraction_summary['successful'] += 1
                else:
                    extraction_summary['failed'] += 1
                    extraction_summary['errors'].append({
                        'field': result['field_name'],
                        'error': result['error']
                    })

            except Exception as e:
                results[field] = None
                page_references[field] = []
                extraction_summary['failed'] += 1
                extraction_summary['errors'].append({
                    'field': field,
                    'error': f'Exception: {str(e)}'
                })

    # Normalize metadata
    normalized = normalize_metadata(results)
    normalized['_extraction_summary'] = extraction_summary
    normalized['_page_references'] = page_references

    return normalized


def extract_metadata(document_text: str, model: str = None, pages_text: List[Tuple[int, str]] = None) -> dict:
    """
    Legacy function - now calls extract_metadata_parallel.
    Kept for backward compatibility.

    Args:
        document_text: Full document text
        model: Model to use
        pages_text: List of (page_number, text) tuples for page references
    """
    return extract_metadata_parallel(document_text, model, pages_text=pages_text)


def normalize_metadata(metadata: dict) -> dict:
    """Normalize and validate extracted metadata"""

    normalized = {
        'irb_number': None,
        'principal_investigator': None,
        'study_title': None,
        'study_purpose': None,
        'inclusion_criteria': None,
        'exclusion_criteria': None,
        'data_elements': None,
        'funding_source': None,
        'protocol_status': None,
        'expiration_date': None
    }

    for key in normalized.keys():
        value = metadata.get(key)

        if not value or str(value).strip().lower() in ['not specified', 'n/a', 'none', 'null']:
            normalized[key] = None
            continue

        value = str(value).strip()

        # Special handling for protocol_status
        if key == 'protocol_status':
            value = value.lower()
            if value not in ['approved', 'pending', 'expired', 'exempt', 'withdrawn']:
                value = None

        # Special handling for dates
        if key == 'expiration_date':
            value = parse_date(value)

        normalized[key] = value

    return normalized


def parse_date(date_string: str) -> Optional[date]:
    """
    Parse various date formats to a Python date object.
    Returns None when parsing fails.
    """
    if date_string is None:
        return None

    if isinstance(date_string, datetime):
        return date_string.date()
    if isinstance(date_string, date):
        return date_string

    value = str(date_string).strip()
    if not value:
        return None

    # If a range is provided like "1/1/2024 - 12/31/2024", take the end date
    if " - " in value:
        last_part = value.split("-")[-1].strip()
        parsed = parse_date(last_part)
        if parsed:
            return parsed

    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%Y/%m/%d',
        '%m-%d-%Y',
        '%d-%m-%Y'
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    # Handle month/year only (e.g., "December 2024" or "12/2024")
    month_year_match = re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{4})$', value, re.IGNORECASE)
    if month_year_match:
        month_name, year_str = month_year_match.groups()
        try:
            try:
                month_num = datetime.strptime(month_name, '%B').month
            except ValueError:
                month_num = datetime.strptime(month_name, '%b').month
            year_num = int(year_str)
            last_day = calendar.monthrange(year_num, month_num)[1]
            return datetime(year_num, month_num, last_day).date()
        except Exception:
            return None

    numeric_month_year = re.match(r'^(\d{1,2})[/-](\d{4})$', value)
    if numeric_month_year:
        month_num = int(numeric_month_year.group(1))
        year_num = int(numeric_month_year.group(2))
        if 1 <= month_num <= 12:
            last_day = calendar.monthrange(year_num, month_num)[1]
            return datetime(year_num, month_num, last_day).date()

    return None


# API endpoint function for single-field extraction
def extract_field_api(field_name: str, document_text: str) -> Dict:
    """
    API-friendly function for extracting a single field.
    Can be called independently for each field.

    Usage:
        result = extract_field_api('irb_number', document_text)
        # Returns: {'field': 'irb_number', 'value': 'IRB-2024-001', 'status': 'success'}
    """
    if field_name not in FIELD_PROMPTS:
        return {
            'field': field_name,
            'value': None,
            'status': 'error',
            'message': f'Unknown field: {field_name}. Valid fields: {list(FIELD_PROMPTS.keys())}'
        }

    result = extract_single_field(field_name, document_text)

    return {
        'field': result['field_name'],
        'value': result['value'],
        'status': result['status'],
        'message': result['error'] if result['status'] == 'failed' else 'Success'
    }
