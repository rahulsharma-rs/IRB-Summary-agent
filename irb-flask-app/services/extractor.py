import calendar
import json
import re
import asyncio
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Field-specific extraction prompts - IMPROVED VERSION
FIELD_PROMPTS = {
    "irb_number": {
        "system": "You are an expert at identifying IRB protocol numbers. Extract ONLY the official IRB protocol identifier.",
        "user_template": """Find the IRB protocol number in this document.

IMPORTANT: Look for the OFFICIAL IRB protocol identifier, typically in one of these formats:
- IRB-YYYYNNNNN (e.g., IRB-300006637)
- IRB YYYYNNNNN
- Protocol #NNNNN
- IRB#NNNNN

Common locations:
- Header or footer of pages
- "IRB Protocol Number:" field
- "ePortfolio" or "IRB ePortfolio" sections
- Protocol approval stamps

Document excerpt:
{text}

RULES:
1. Extract the EXACT protocol number as written
2. Include all digits and formatting (dashes, spaces)
3. If multiple numbers appear, choose the one labeled as "IRB Protocol Number" or "IRB Number"
4. Do NOT include version numbers or amendment numbers

Return ONLY valid JSON:
{{"irb_number": "IRB-300006637"}}

If no IRB number found: {{"irb_number": null}}""",
        "max_tokens": 150,
        "search_keywords": ["irb", "protocol", "eportfolio", "protocol number", "irb number", "irb#"]
    },

    "principal_investigator": {
        "system": "You are an expert at identifying Principal Investigators. Extract the PI's full name exactly as written.",
        "user_template": """Find the Principal Investigator (PI) name in this document.

Look for these exact labels:
- "Principal Investigator:"
- "PI:"
- "Lead Investigator:"
- "Primary Investigator:"

Common locations:
- Top of first page
- Investigator section
- Signature blocks
- Contact information section

Document excerpt:
{text}

RULES:
1. Extract the COMPLETE name including titles (Dr., MD, PhD, etc.)
2. If multiple investigators listed, extract the PRIMARY or PRINCIPAL investigator only
3. Format: "Dr. FirstName LastName" or "FirstName LastName, Credentials"
4. Do NOT include email, phone, or department

Return ONLY valid JSON:
{{"principal_investigator": "Dr. Jane Smith, MD"}}

If not found: {{"principal_investigator": null}}""",
        "max_tokens": 200,
        "search_keywords": ["principal investigator", "pi:", "lead investigator", "investigator:", "study director"]
    },

    "study_title": {
        "system": "You are an expert at identifying official study titles. Extract the complete, formal study title.",
        "user_template": """Find the OFFICIAL study title in this document.

Look for these labels:
- "Study Title:"
- "Project Title:"
- "Research Title:"
- "Protocol Title:"
- "Title:"

Common locations:
- First page, near top
- After IRB number
- In "Study Information" section

Document excerpt:
{text}

RULES:
1. Extract the FULL official title (may be long)
2. Do NOT truncate or summarize
3. Include all subtitles or phases if present
4. Remove any surrounding quotes or formatting
5. This is the title that appears on IRB approval documents

Return ONLY valid JSON:
{{"study_title": "Complete Study Title Here"}}

If not found: {{"study_title": null}}""",
        "max_tokens": 300,
        "search_keywords": ["study title", "project title", "protocol title", "title:", "research title"]
    },

    "study_purpose": {
        "system": "You are an expert at summarizing research purposes. Create a clear 1-2 sentence summary.",
        "user_template": """Summarize the study purpose or objectives in 1-2 clear sentences.

Look for sections labeled:
- "Purpose"
- "Objectives"
- "Study Aims"
- "Research Question"
- "Background"
- "Rationale"

Document excerpt:
{text}

RULES:
1. Write in YOUR OWN WORDS - do not copy verbatim
2. Keep it to 1-2 sentences maximum
3. Focus on: What is being studied and why?
4. Make it understandable to non-experts
5. Example: "This study aims to evaluate the effectiveness of X in treating Y among Z population."

Return ONLY valid JSON:
{{"study_purpose": "Brief 1-2 sentence summary here"}}

If not found: {{"study_purpose": null}}""",
        "max_tokens": 400,
        "search_keywords": ["purpose", "objective", "aims", "background", "research question", "rationale",
                            "hypothesis"]
    },

    "inclusion_criteria": {
        "system": "You are an expert at extracting inclusion criteria. List the key eligibility requirements.",
        "user_template": """Extract the KEY inclusion criteria (who CAN participate).

Look for section labeled:
- "Inclusion Criteria"
- "Eligibility Criteria"
- "Subject Selection Criteria"

Document excerpt:
{text}

RULES:
1. List ONLY the major inclusion criteria (3-5 most important)
2. Use bullet points format
3. Be concise but complete
4. Example format:
   • Age 18-65 years
   • Diagnosed with condition X
   • Able to provide consent

Return ONLY valid JSON:
{{"inclusion_criteria": "• Criterion 1\\n• Criterion 2\\n• Criterion 3"}}

If not found: {{"inclusion_criteria": null}}""",
        "max_tokens": 500,
        "search_keywords": ["inclusion criteria", "eligibility", "eligible", "subject selection",
                            "participant criteria"]
    },

    "exclusion_criteria": {
        "system": "You are an expert at extracting exclusion criteria. List the key disqualifying factors.",
        "user_template": """Extract the KEY exclusion criteria (who CANNOT participate).

Look for section labeled:
- "Exclusion Criteria"
- "Ineligibility Criteria"

Document excerpt:
{text}

RULES:
1. List ONLY the major exclusion criteria (3-5 most important)
2. Use bullet points format
3. Be concise but complete
4. Example format:
   • Pregnant or breastfeeding
   • History of condition Y
   • Unable to provide consent

Return ONLY valid JSON:
{{"exclusion_criteria": "• Criterion 1\\n• Criterion 2\\n• Criterion 3"}}

If not found: {{"exclusion_criteria": null}}""",
        "max_tokens": 500,
        "search_keywords": ["exclusion criteria", "ineligibility", "not eligible", "cannot participate", "exclusions"]
    },

    "data_elements": {
        "system": "You are an expert at identifying approved data elements and PHI identifiers.",
        "user_template": """Find what data elements, identifiers, or PHI are approved for collection/use.

Look for sections like:
- "Data Elements"
- "Data to be Collected"
- "PHI Requested"
- "Identifiers"
- "Data Requested"
- "Variables"

Document excerpt:
{text}

RULES:
1. List specific data elements if enumerated
2. Include any date ranges mentioned
3. Include any identifiers (names, MRNs, dates, etc.)
4. Be specific but concise
5. Example: "Demographics, diagnoses, lab results, treatment dates (2020-2023), patient names and MRNs"

Return ONLY valid JSON:
{{"data_elements": "List of data elements and date ranges"}}

If not found: {{"data_elements": null}}""",
        "max_tokens": 400,
        "search_keywords": ["data elements", "identifiers", "phi", "data requested", "date range", "data collection",
                            "variables"]
    },

    "funding_source": {
        "system": "You are an expert at identifying funding sources and sponsors.",
        "user_template": """Find the funding source(s) or sponsor(s) for this study.

Look for sections:
- "Funding Source"
- "Sponsor"
- "Grant Number"
- "Financial Support"
- "Funded by"

Document excerpt:
{text}

RULES:
1. Extract the organization/entity providing funding
2. Include grant numbers if mentioned
3. If "None" or "Unfunded", state that clearly
4. Example: "National Institutes of Health (NIH Grant R01-123456)"
5. Example: "Unfunded" or "University internal funding"

Return ONLY valid JSON:
{{"funding_source": "Funding organization or 'Unfunded'"}}

If not found: {{"funding_source": null}}""",
        "max_tokens": 300,
        "search_keywords": ["funding", "sponsor", "grant", "support", "financial", "funded by", "nih", "contract"]
    },

    "protocol_status": {
        "system": "You are an expert at identifying protocol approval status. Use ONLY the allowed status values.",
        "user_template": """Find the current protocol approval status.

ALLOWED STATUS VALUES (use lowercase):
- approved
- pending
- expired
- exempt
- withdrawn

Look for phrases like:
- "Status: Approved"
- "Exempt determination"
- "IRB Approval"
- "Expired"
- "Pending review"

Document excerpt:
{text}

RULES:
1. Return EXACTLY ONE of the allowed values (lowercase)
2. "Approved" = IRB has approved the protocol
3. "Exempt" = Determined to be exempt from full review
4. "Expired" = Previous approval has expired
5. "Pending" = Awaiting IRB decision
6. "Withdrawn" = Protocol was withdrawn

Return ONLY valid JSON:
{{"protocol_status": "approved"}}

If not found: {{"protocol_status": null}}""",
        "max_tokens": 150,
        "search_keywords": ["status", "approval", "exempt", "expired", "withdrawn", "pending", "determination"]
    },

    "expiration_date": {
        "system": "You are an expert at finding protocol expiration or approval dates. Return dates in YYYY-MM-DD format.",
        "user_template": """Find the protocol expiration date or approval period end date.

Look for labels:
- "Expiration Date:"
- "Approval Expires:"
- "Valid Until:"
- "Continuing Review Due:"
- "Approval Period:"

Document excerpt:
{text}

RULES:
1. Return date in YYYY-MM-DD format (e.g., 2025-12-31)
2. If only month/year given, use last day of month
3. If "Approval Period: 1/1/2024 - 12/31/2024", extract the END date
4. Convert formats: "December 31, 2024" → "2024-12-31"

Return ONLY valid JSON:
{{"expiration_date": "2025-12-31"}}

If not found: {{"expiration_date": null}}""",
        "max_tokens": 150,
        "search_keywords": ["expiration", "expires", "approval date", "valid until", "continuing review",
                            "approval period"]
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
        prompt_config['search_keywords']
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
