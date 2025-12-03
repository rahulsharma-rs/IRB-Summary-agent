import os
from pathlib import Path


class Config:
    # Base directory
    BASE_DIR = Path(__file__).parent

    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    APPLICATION_ROOT = os.environ.get('APPLICATION_ROOT', '/irb')
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        f'sqlite:///{BASE_DIR / "irb_documents.db"}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5-nano')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')

    # Upload settings
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

    # RAG settings
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 20

    # Extraction prompt
    EXTRACTION_PROMPT = """You are an expert IRB protocol metadata extractor. Analyze the provided IRB document and extract ONLY the following information in JSON format:

{
  "irb_number": "The IRB protocol number (e.g., IRB-2024-001)",
  "principal_investigator": "Name(s) of PI(s)",
  "study_title": "Full official study title",
  "study_purpose": "1-2 sentence summary of study purpose/objectives",
  "inclusion_criteria": "Key inclusion criteria (bullet points or brief text)",
  "exclusion_criteria": "Key exclusion criteria (bullet points or brief text)",
  "data_elements": "Approved data elements, identifiers, date ranges",
  "funding_source": "Funding sources or sponsors (if specified)",
  "protocol_status": "approved | pending | expired | exempt | withdrawn",
  "expiration_date": "Protocol expiration date (YYYY-MM-DD format if found)"
}

CRITICAL RULES:
1. Extract information EXACTLY as it appears in the document
2. Use "Not specified" for missing fields
3. For dates, use ISO format (YYYY-MM-DD) or "Not specified"
4. For protocol_status, use only the exact values listed above
5. Return ONLY valid JSON, no additional text
6. Be thorough but concise - max 200 chars per field except criteria fields

Document text:
{document_text}

Return only the JSON object."""

    @staticmethod
    def init_app(app):
        """Initialize application folders"""
        Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
