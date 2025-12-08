from sqlalchemy import or_, and_, func
from models.database import Document, db
from datetime import datetime
import numpy as np
from services.embedder import embed_single


def create_document_search_profile(doc: Document) -> str:
    """
    Create a rich text profile of document metadata for embedding.
    This profile will be embedded and used for semantic search.

    Args:
        doc: Document object with extracted metadata

    Returns:
        Combined text profile of all metadata fields
    """
    parts = []

    if doc.irb_number:
        parts.append(f"IRB Number: {doc.irb_number}")

    if doc.principal_investigator:
        parts.append(f"Principal Investigator: {doc.principal_investigator}")

    if doc.study_title:
        parts.append(f"Study Title: {doc.study_title}")

    if doc.study_purpose:
        parts.append(f"Study Purpose: {doc.study_purpose}")

    if doc.inclusion_criteria:
        parts.append(f"Inclusion Criteria: {doc.inclusion_criteria}")

    if doc.exclusion_criteria:
        parts.append(f"Exclusion Criteria: {doc.exclusion_criteria}")

    if doc.data_elements:
        parts.append(f"Data Elements: {doc.data_elements}")

    if doc.funding_source:
        parts.append(f"Funding Source: {doc.funding_source}")

    if doc.protocol_status:
        parts.append(f"Protocol Status: {doc.protocol_status}")

    return "\n".join(parts)


def semantic_search_documents(query: str,
                              status: str = None,
                              date_from: str = None,
                              date_to: str = None,
                              top_k: int = 50,
                              similarity_threshold: float = 0.0) -> list:
    """
    Semantic search using document-level embeddings.

    Args:
        query: Natural language search query
        status: Optional protocol status filter
        date_from: Optional start date filter (YYYY-MM-DD)
        date_to: Optional end date filter (YYYY-MM-DD)
        top_k: Maximum number of results
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of (Document, similarity_score) tuples, sorted by similarity
    """

    # Get all successfully extracted documents with embeddings
    base_query = Document.query.filter(
        Document.extraction_status.in_(['success', 'partial'])
    ).filter(
        Document.document_embedding.isnot(None),
        Document.document_embedding != ''
    )

    # Apply optional filters
    filters = []

    if status and status.strip():
        filters.append(Document.protocol_status == status.strip().lower())

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

    if filters:
        base_query = base_query.filter(and_(*filters))

    # Get all documents that match filters
    documents = base_query.all()
    print(f"[SEMANTIC_SEARCH] Query='{query}' top_k={top_k} threshold={similarity_threshold}")
    print(f"[SEMANTIC_SEARCH] Candidates after filters: {len(documents)}")

    if not documents:
        return []

    # Embed the query
    try:
        query_embedding = embed_single(query)
        print(f"[SEMANTIC_SEARCH] Query embedding length: {len(query_embedding) if query_embedding is not None else 0}")
    except Exception as e:
        print(f"[SEMANTIC_SEARCH] Failed to embed query: {e}")
        return []

    # Calculate similarities
    results = []
    for doc in documents:
        doc_embedding = doc.get_document_embedding()
        if doc_embedding is None:
            print(f"[SEMANTIC_SEARCH] Skipping doc {doc.id} (no embedding)")
            continue

        # Cosine similarity (embeddings are already normalized)
        similarity = float(np.dot(query_embedding, doc_embedding))

        if similarity >= similarity_threshold:
            results.append((doc, similarity))
        else:
            # Still keep, just at low score if threshold is 0
            if similarity_threshold <= 0:
                results.append((doc, similarity))
        if similarity >= similarity_threshold:
            print(f"[SEMANTIC_SEARCH] Doc {doc.id} similarity {similarity:.3f}")

    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    # Slice and log top results
    top_results = results[:top_k]
    print("[SEMANTIC_SEARCH] Top results:")
    for rank, (doc, score) in enumerate(top_results, 1):
        print(f"  #{rank} doc_id={doc.id} file={doc.filename} score={score:.3f}")

    return top_results


def search_documents(query: str = None,
                     irb_number: str = None,
                     pi_name: str = None,
                     status: str = None,
                     date_from: str = None,
                     date_to: str = None,
                     limit: int = 50) -> list:
    """
    Traditional keyword-based search using SQL LIKE/ILIKE.

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
                Document.irb_number.ilike(search_term),
                Document.inclusion_criteria.ilike(search_term),
                Document.exclusion_criteria.ilike(search_term),
                Document.data_elements.ilike(search_term)
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
