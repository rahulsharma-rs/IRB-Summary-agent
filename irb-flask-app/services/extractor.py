import json
import re
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Field-specific extraction prompts
FIELD_PROMPTS = {
    "irb_number": {
        "system": "You are an expert at identifying IRB protocol numbers in regulatory documents.",
        "user_template": """Find the IRB protocol number in this document text.

IRB numbers typically look like:
- IRB-2024-001
- Protocol #12345
- IRB20240001
- 2024-IRB-123

Document excerpt:
{text}

Return ONLY a JSON object:
{{"irb_number": "the protocol number"}}

If not found, return: {{"irb_number": null}}
Do not include any other text.""",
        "max_tokens": 150,
        "search_keywords": ["irb", "protocol number", "protocol #", "protocol id"]
    },

    "principal_investigator": {
        "system": "You are an expert at identifying Principal Investigators in research documents.",
        "user_template": """Find the Principal Investigator(s) name(s) in this document.

Look for labels like:
- Principal Investigator
- PI:
- Lead Investigator
- Study Director

Document excerpt:
{text}

Return ONLY a JSON object:
{{"principal_investigator": "Full Name(s)"}}

If multiple PIs, separate with semicolons.
If not found, return: {{"principal_investigator": null}}
Do not include any other text.""",
        "max_tokens": 200,
        "search_keywords": ["principal investigator", "pi:", "lead investigator", "study director"]
    },

    "study_title": {
        "system": "You are an expert at identifying study titles in research protocols.",
        "user_template": """Find the official study title in this document.

Look for labels like:
- Study Title
- Project Title
- Title:
- Research Title

Document excerpt:
{text}

Return ONLY a JSON object:
{{"study_title": "The full official study title"}}

If not found, return: {{"study_title": null}}
Do not include any other text.""",
        "max_tokens": 300,
        "search_keywords": ["study title", "project title", "title:", "research title"]
    },

    "study_purpose": {
        "system": "You are an expert at summarizing research study purposes and objectives.",
        "user_template": """Summarize the study purpose/objectives in 1-2 sentences.

Look for sections like:
- Purpose
- Objectives
- Background
- Study Aims
- Research Question

Document excerpt:
{text}

Return ONLY a JSON object:
{{"study_purpose": "1-2 sentence summary"}}

If not found, return: {{"study_purpose": null}}
Do not include any other text.""",
        "max_tokens": 400,
        "search_keywords": ["purpose", "objective", "aims", "background", "research question"]
    },

    "inclusion_criteria": {
        "system": "You are an expert at identifying inclusion criteria in clinical protocols.",
        "user_template": """Extract the key inclusion criteria for study participants.

Look for sections like:
- Inclusion Criteria
- Eligibility Criteria
- Subject Selection

Document excerpt:
{text}

Return ONLY a JSON object:
{{"inclusion_criteria": "Brief bullet-point list or concise summary"}}

Keep it under 200 words.
If not found, return: {{"inclusion_criteria": null}}
Do not include any other text.""",
        "max_tokens": 500,
        "search_keywords": ["inclusion criteria", "eligibility", "subject selection", "eligible"]
    },

    "exclusion_criteria": {
        "system": "You are an expert at identifying exclusion criteria in clinical protocols.",
        "user_template": """Extract the key exclusion criteria for study participants.

Look for sections like:
- Exclusion Criteria
- Ineligibility Criteria

Document excerpt:
{text}

Return ONLY a JSON object:
{{"exclusion_criteria": "Brief bullet-point list or concise summary"}}

Keep it under 200 words.
If not found, return: {{"exclusion_criteria": null}}
Do not include any other text.""",
        "max_tokens": 500,
        "search_keywords": ["exclusion criteria", "ineligibility", "not eligible", "cannot participate"]
    },

    "data_elements": {
        "system": "You are an expert at identifying approved data elements in IRB protocols.",
        "user_template": """Find what data elements, identifiers, or date ranges are approved for this study.

Look for sections like:
- Data Elements
- Data Collection
- PHI/Identifiers
- Data Requested
- Date Range

Document excerpt:
{text}

Return ONLY a JSON object:
{{"data_elements": "List of approved data elements, identifiers, date ranges"}}

If not found, return: {{"data_elements": null}}
Do not include any other text.""",
        "max_tokens": 400,
        "search_keywords": ["data elements", "identifiers", "phi", "data requested", "date range", "data collection"]
    },

    "funding_source": {
        "system": "You are an expert at identifying funding sources and sponsors in research documents.",
        "user_template": """Find the funding source(s) or sponsor(s) for this study.

Look for sections like:
- Funding Source
- Sponsor
- Grant
- Financial Support

Document excerpt:
{text}

Return ONLY a JSON object:
{{"funding_source": "Funding sources/sponsors"}}

If not found, return: {{"funding_source": null}}
Do not include any other text.""",
        "max_tokens": 300,
        "search_keywords": ["funding", "sponsor", "grant", "support", "financial"]
    },

    "protocol_status": {
        "system": "You are an expert at identifying protocol approval status in IRB documents.",
        "user_template": """Find the current protocol status.

Valid statuses are ONLY:
- approved
- pending
- expired
- exempt
- withdrawn

Look for phrases like:
- "Status: Approved"
- "Exempt determination"
- "Approval expired"

Document excerpt:
{text}

Return ONLY a JSON object:
{{"protocol_status": "one of: approved, pending, expired, exempt, withdrawn"}}

Use lowercase. If not found, return: {{"protocol_status": null}}
Do not include any other text.""",
        "max_tokens": 150,
        "search_keywords": ["status", "approval", "exempt", "expired", "withdrawn"]
    },

    "expiration_date": {
        "system": "You are an expert at finding protocol expiration dates in IRB documents.",
        "user_template": """Find the protocol expiration date or approval end date.

Look for phrases like:
- Expiration Date
- Approval Period
- Valid Until
- Continuing Review Due

Document excerpt:
{text}

Return ONLY a JSON object:
{{"expiration_date": "YYYY-MM-DD format or descriptive date"}}

If not found, return: {{"expiration_date": null}}
Do not include any other text.""",
        "max_tokens": 150,
        "search_keywords": ["expiration", "approval date", "valid until", "continuing review", "expires"]
    }
}


def find_relevant_sections(full_text: str, keywords: List[str], context_chars: int = 3000) -> str:
    """
    Find sections of text most relevant to the keywords.
    Returns up to context_chars of the most relevant text.
    """
    text_lower = full_text.lower()

    # Find all keyword positions
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
        # No keywords found, return beginning of document
        return full_text[:context_chars]

    # Find the section with most keyword density
    positions.sort()
    best_start = 0
    best_count = 0

    for start_pos in positions:
        end_pos = start_pos + context_chars
        count = sum(1 for p in positions if start_pos <= p < end_pos)
        if count > best_count:
            best_count = count
            best_start = max(0, start_pos - 500)  # Include some context before

    return full_text[best_start:best_start + context_chars]


def extract_single_field(field_name: str, document_text: str, model: str = None) -> Dict:
    """
    Extract a single field from the document using a focused prompt.

    Returns:
        dict with 'field_name', 'value', 'status' ('success' or 'failed'), 'error'
    """
    if not model:
        model = Config.OPENAI_MODEL

    if field_name not in FIELD_PROMPTS:
        return {
            'field_name': field_name,
            'value': None,
            'status': 'failed',
            'error': f'Unknown field: {field_name}'
        }

    prompt_config = FIELD_PROMPTS[field_name]

    # Find relevant sections in document
    relevant_text = find_relevant_sections(
        document_text,
        prompt_config['search_keywords']
    )

    user_prompt = prompt_config['user_template'].format(text=relevant_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_config['system']},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=prompt_config['max_tokens']
        )

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

        return {
            'field_name': field_name,
            'value': value,
            'status': 'success',
            'error': None
        }

    except json.JSONDecodeError as e:
        return {
            'field_name': field_name,
            'value': None,
            'status': 'failed',
            'error': f'JSON parse error: {str(e)}'
        }
    except Exception as e:
        return {
            'field_name': field_name,
            'value': None,
            'status': 'failed',
            'error': str(e)
        }


def extract_metadata_parallel(document_text: str, model: str = None, max_workers: int = 5) -> dict:
    """
    Extract all metadata fields in parallel using ThreadPoolExecutor.

    This is the NEW recommended approach - extracts each field independently
    with focused prompts, running multiple extractions concurrently.

    Returns:
        dict with all field values and an extraction_summary
    """
    if not model:
        model = Config.OPENAI_MODEL

    fields_to_extract = list(FIELD_PROMPTS.keys())
    results = {}
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
            executor.submit(extract_single_field, field, document_text, model): field
            for field in fields_to_extract
        }

        # Collect results as they complete
        for future in as_completed(future_to_field):
            field = future_to_field[future]
            try:
                result = future.result()
                results[result['field_name']] = result['value']

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
                extraction_summary['failed'] += 1
                extraction_summary['errors'].append({
                    'field': field,
                    'error': f'Exception: {str(e)}'
                })

    # Normalize metadata
    normalized = normalize_metadata(results)
    normalized['_extraction_summary'] = extraction_summary

    return normalized


def extract_metadata(document_text: str, model: str = None) -> dict:
    """
    Legacy function - now calls extract_metadata_parallel.
    Kept for backward compatibility.
    """
    return extract_metadata_parallel(document_text, model)


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


def parse_date(date_string: str) -> str:
    """Parse various date formats to ISO format (YYYY-MM-DD)"""
    if not date_string:
        return None

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
            dt = datetime.strptime(date_string, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue

    return date_string


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