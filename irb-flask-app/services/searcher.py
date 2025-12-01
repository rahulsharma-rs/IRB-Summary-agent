from sqlalchemy import or_, and_, func
from models.database import Document, db
from datetime import datetime


def search_documents(query: str = None,
                     irb_number: str = None,
                     pi_name: str = None,
                     status: str = None,
                     date_from: str = None,
                     date_to: str = None,
                     limit: int = 50) -> list:
    """
    Search documents with multiple filters

    Args:
        query: General text search across title, purpose, PI
        irb_number: Filter by IRB number
        pi_name: Filter by PI name
        status: Filter by protocol status
        date_from: Filter by upload date (YYYY-MM-DD)
        date_to: Filter by upload date (YYYY-MM-DD)
        limit: Maximum results to return

    Returns:
        List of Document objects
    """

    # Start with base query
    q = Document.query

    # Apply filters
    filters = []

    # General text search (case-insensitive)
    if query and query.strip():
        search_term = f"%{query.strip()}%"
        filters.append(
            or_(
                Document.study_title.ilike(search_term),
                Document.study_purpose.ilike(search_term),
                Document.principal_investigator.ilike(search_term),
                Document.irb_number.ilike(search_term)
            )
        )

    # IRB number filter
    if irb_number and irb_number.strip():
        filters.append(Document.irb_number.ilike(f"%{irb_number.strip()}%"))

    # PI name filter
    if pi_name and pi_name.strip():
        filters.append(Document.principal_investigator.ilike(f"%{pi_name.strip()}%"))

    # Status filter
    if status and status.strip():
        filters.append(Document.protocol_status == status.strip().lower())

    # Date range filters
    if date_from:
        try:
            dt_from = datetime.strptime(date_from, '%Y-%m-%d')
            filters.append(Document.upload_date >= dt_from)
        except ValueError:
            pass

    if date_to:
        try:
            dt_to = datetime.strptime(date_to, '%Y-%m-%d')
            filters.append(Document.upload_date <= dt_to)
        except ValueError:
            pass

    # Apply all filters
    if filters:
        q = q.filter(and_(*filters))

    # Only show successfully extracted documents (or partial if they have enough data)
    q = q.filter(Document.extraction_status.in_(['success', 'partial']))

    # Order by upload date (newest first)
    q = q.order_by(Document.upload_date.desc())

    # Apply limit
    results = q.limit(limit).all()

    return results


def get_document_stats() -> dict:
    """Get database statistics"""

    total_docs = Document.query.count()

    status_counts = db.session.query(
        Document.protocol_status,
        func.count(Document.id)
    ).group_by(Document.protocol_status).all()

    by_status = {status: count for status, count in status_counts if status}

    return {
        'total_documents': total_docs,
        'by_status': by_status,
        'total_chunks': db.session.query(func.count()).select_from(
            db.session.query(Document.id).join(Document.chunks).subquery()
        ).scalar() or 0
    }