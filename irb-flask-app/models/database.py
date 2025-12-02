from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index, text

db = SQLAlchemy()


class Document(db.Model):
    __tablename__ = 'documents'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    file_size = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Extracted metadata
    irb_number = db.Column(db.String(100), index=True)
    irb_number_pages = db.Column(db.String(100))  # Comma-separated page numbers
    principal_investigator = db.Column(db.String(255), index=True)
    principal_investigator_pages = db.Column(db.String(100))
    study_title = db.Column(db.Text)
    study_title_pages = db.Column(db.String(100))
    study_purpose = db.Column(db.Text)
    study_purpose_pages = db.Column(db.String(100))
    inclusion_criteria = db.Column(db.Text)
    inclusion_criteria_pages = db.Column(db.String(100))
    exclusion_criteria = db.Column(db.Text)
    exclusion_criteria_pages = db.Column(db.String(100))
    data_elements = db.Column(db.Text)
    data_elements_pages = db.Column(db.String(100))
    funding_source = db.Column(db.String(500))
    funding_source_pages = db.Column(db.String(100))
    protocol_status = db.Column(db.String(50), index=True)
    protocol_status_pages = db.Column(db.String(100))
    expiration_date = db.Column(db.Date)
    expiration_date_pages = db.Column(db.String(100))

    # Processing metadata
    n_pages = db.Column(db.Integer)
    n_chunks = db.Column(db.Integer)
    extraction_status = db.Column(db.String(50), default='pending')  # pending, success, partial, failed
    extraction_error = db.Column(db.Text)

    # Track if manually edited
    manually_edited = db.Column(db.Boolean, default=False)
    last_edited_date = db.Column(db.DateTime)

    # Relationships
    chunks = db.relationship('DocumentChunk', backref='document',
                             lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""

        def parse_pages(page_str):
            """Convert comma-separated page string to list of ints"""
            if not page_str:
                return []
            try:
                return [int(p.strip()) for p in page_str.split(',') if p.strip()]
            except:
                return []

        return {
            'id': self.id,
            'filename': self.filename,
            'file_hash': self.file_hash,
            'upload_date': self.upload_date.isoformat(),
            'irb_number': self.irb_number,
            'irb_number_pages': parse_pages(self.irb_number_pages),
            'principal_investigator': self.principal_investigator,
            'principal_investigator_pages': parse_pages(self.principal_investigator_pages),
            'study_title': self.study_title,
            'study_title_pages': parse_pages(self.study_title_pages),
            'study_purpose': self.study_purpose,
            'study_purpose_pages': parse_pages(self.study_purpose_pages),
            'inclusion_criteria': self.inclusion_criteria,
            'inclusion_criteria_pages': parse_pages(self.inclusion_criteria_pages),
            'exclusion_criteria': self.exclusion_criteria,
            'exclusion_criteria_pages': parse_pages(self.exclusion_criteria_pages),
            'data_elements': self.data_elements,
            'data_elements_pages': parse_pages(self.data_elements_pages),
            'funding_source': self.funding_source,
            'funding_source_pages': parse_pages(self.funding_source_pages),
            'protocol_status': self.protocol_status,
            'protocol_status_pages': parse_pages(self.protocol_status_pages),
            'expiration_date': self.expiration_date.isoformat() if self.expiration_date else None,
            'expiration_date_pages': parse_pages(self.expiration_date_pages),
            'n_pages': self.n_pages,
            'n_chunks': self.n_chunks,
            'extraction_status': self.extraction_status
        }


class DocumentChunk(db.Model):
    __tablename__ = 'document_chunks'

    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False, index=True)
    page_number = db.Column(db.Integer)
    chunk_index = db.Column(db.Integer)
    text = db.Column(db.Text, nullable=False)

    # Embedding stored as comma-separated float string (SQLite doesn't have array type)
    embedding = db.Column(db.Text)

    __table_args__ = (
        Index('idx_doc_page', 'document_id', 'page_number'),
        Index('idx_doc_chunk', 'document_id', 'chunk_index'),
    )

    def get_embedding(self):
        """Convert stored embedding string back to list of floats"""
        if not self.embedding:
            return None
        return [float(x) for x in self.embedding.split(',')]

    def set_embedding(self, embedding_array):
        """Store embedding array as comma-separated string"""
        self.embedding = ','.join(str(x) for x in embedding_array)


def init_db(app):
    """Initialize database"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database initialized successfully")