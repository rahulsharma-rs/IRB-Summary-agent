import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime

from config import Config
from models.database import db, init_db, Document, DocumentChunk
from models.document import (
    extract_pages, chunk_text, compute_file_hash, get_full_text
)
from services.extractor import extract_metadata, parse_date
from services.embedder import embed_texts, embed_single, find_similar_chunks
from services.searcher import search_documents, get_document_stats

app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Set application root for URL generation
app.config['APPLICATION_ROOT'] = '/irb'

# Fix URL generation for proxy setup
from flask import _request_ctx_stack


@app.url_defaults
def add_prefix(endpoint, values):
    if 'static' in endpoint:
        return


@app.url_value_preprocessor
def pull_prefix(endpoint, values):
    pass
# Initialize database
init_db(app)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page with upload form"""
    stats = get_document_stats()
    return render_template('index.html', stats=stats)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Read file
        file_bytes = file.read()
        file_hash = compute_file_hash(file_bytes)

        # Check if already exists
        existing = Document.query.filter_by(file_hash=file_hash).first()
        if existing:
            return jsonify({
                'message': 'Document already exists',
                'document_id': existing.id,
                'redirect': url_for('document_detail', doc_id=existing.id)
            }), 200

        # Extract pages
        pages = extract_pages(file.filename, file_bytes)
        full_text = get_full_text(pages)

        # Chunk and embed early so extraction can use stored embeddings
        chunks = chunk_text(pages,
                            max_chars=Config.CHUNK_SIZE,
                            overlap=Config.CHUNK_OVERLAP)
        chunk_embeddings = None
        chunk_texts = []
        if chunks:
            chunk_texts = [c['text'] for c in chunks]
            chunk_embeddings = embed_texts(chunk_texts)

        # Extract metadata using parallel extraction WITH page tracking
        try:
            print(f"[EXTRACTION] Starting parallel extraction for {file.filename}")
            metadata = extract_metadata(
                full_text,
                pages_text=pages,
                chunk_texts=chunk_texts if chunk_texts else None,
                chunk_embeddings=chunk_embeddings
            )
            exp_date = metadata.get('expiration_date')
            if isinstance(exp_date, str):
                metadata['expiration_date'] = parse_date(exp_date)

            # Check extraction summary
            summary = metadata.pop('_extraction_summary', {})
            page_refs = metadata.pop('_page_references', {})
            successful = summary.get('successful', 0)
            failed = summary.get('failed', 0)

            print(f"[EXTRACTION] Completed: {successful} successful, {failed} failed")

            if failed > 0:
                print(f"[EXTRACTION] Errors: {summary.get('errors', [])}")

            # Consider it a success if at least half the fields were extracted
            if successful >= 5:
                extraction_status = 'success'
                extraction_error = None
            else:
                extraction_status = 'partial'
                extraction_error = f"Only {successful}/10 fields extracted successfully"

        except Exception as e:
            print(f"[EXTRACTION] Fatal error: {str(e)}")
            metadata = {}
            page_refs = {}
            extraction_status = 'failed'
            extraction_error = str(e)

        # Create document record
        doc = Document(
            filename=secure_filename(file.filename),
            file_hash=file_hash,
            file_size=len(file_bytes),
            n_pages=len(pages),
            extraction_status=extraction_status,
            extraction_error=extraction_error,
            **metadata
        )

        # Store page references
        for field_name, page_list in page_refs.items():
            if page_list:
                pages_str = ','.join(str(p) for p in page_list)
                setattr(doc, f"{field_name}_pages", pages_str)

        db.session.add(doc)
        db.session.flush()  # Get document ID

        # Store chunks with precomputed embeddings
        if chunks:
            for chunk_data, embedding in zip(chunks, chunk_embeddings):
                chunk = DocumentChunk(
                    document_id=doc.id,
                    page_number=chunk_data['page_number'],
                    chunk_index=chunk_data['chunk_index'],
                    text=chunk_data['text']
                )
                chunk.set_embedding(embedding)
                db.session.add(chunk)

            doc.n_chunks = len(chunks)

        db.session.commit()

        return jsonify({
            'message': 'Document uploaded successfully',
            'document_id': doc.id,
            'redirect': url_for('document_detail', doc_id=doc.id)
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/search')
def search():
    """Search page"""
    query = request.args.get('q', '')
    irb_number = request.args.get('irb', '')
    pi_name = request.args.get('pi', '')
    status = request.args.get('status', '')

    results = []
    if any([query, irb_number, pi_name, status]):
        results = search_documents(
            query=query,
            irb_number=irb_number,
            pi_name=pi_name,
            status=status
        )

    return render_template('search.html',
                           results=results,
                           query=query,
                           irb_number=irb_number,
                           pi_name=pi_name,
                           status=status)


@app.route('/document/<int:doc_id>')
def document_detail(doc_id):
    """Document detail page with RAG query interface"""
    doc = Document.query.get_or_404(doc_id)
    return render_template('document.html', document=doc)


@app.route('/api/query/<int:doc_id>', methods=['POST'])
def query_document(doc_id):
    """RAG query endpoint for a specific document"""

    doc = Document.query.get_or_404(doc_id)
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({'error': 'Question required'}), 400

    question = data['question'].strip()
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400

    try:
        # Get query embedding
        query_embedding = embed_single(question)

        # Get document chunks
        chunks = doc.chunks.all()
        if not chunks:
            return jsonify({'error': 'No chunks available for this document'}), 400

        # Get chunk embeddings
        chunk_embeddings = [chunk.get_embedding() for chunk in chunks]

        # Find similar chunks
        top_k = data.get('top_k', Config.TOP_K_RETRIEVAL)
        similar = find_similar_chunks(query_embedding, chunk_embeddings, top_k=top_k)

        # Prepare context
        context_blocks = []
        for idx, score in similar:
            chunk = chunks[idx]
            context_blocks.append({
                'page': chunk.page_number,
                'text': chunk.text,
                'score': score
            })

        # Generate answer using GPT
        from openai import OpenAI
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        context = "\n\n".join([f"(Page {b['page']}) {b['text']}" for b in context_blocks])

        model_name = Config.OPENAI_MODEL
        model_lower = model_name.lower()
        is_restricted_model = any(x in model_lower for x in ['gpt-5', 'o1', 'o3'])

        chat_params = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an IRB protocol analysis assistant. Answer questions based only on the provided context. Cite page numbers in your answer."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}"
                }
            ]
        }

        if not is_restricted_model:
            chat_params["max_tokens"] = 400
            chat_params["temperature"] = 0.2

        response = client.chat.completions.create(**chat_params)

        answer = response.choices[0].message.content

        return jsonify({
            'answer': answer,
            'context': context_blocks
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents')
def list_documents():
    """API endpoint to list all documents"""
    docs = Document.query.filter_by(extraction_status='success') \
        .order_by(Document.upload_date.desc()) \
        .all()
    return jsonify([doc.to_dict() for doc in docs])


@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    return jsonify(get_document_stats())


@app.route('/api/extract-field/<int:doc_id>', methods=['POST'])
def extract_field_endpoint(doc_id):
    """
    API endpoint to extract or re-extract a single field from a document.

    POST /api/extract-field/<doc_id>
    Body: {"field": "irb_number"}

    Returns: {"field": "irb_number", "value": "IRB-2024-001", "status": "success"}
    """
    doc = Document.query.get_or_404(doc_id)
    data = request.get_json()

    if not data or 'field' not in data:
        return jsonify({'error': 'Field name required'}), 400

    field_name = data['field']

    # Valid fields
    valid_fields = [
        'irb_number', 'principal_investigator', 'study_title',
        'study_purpose', 'inclusion_criteria', 'exclusion_criteria',
        'data_elements', 'funding_source', 'protocol_status', 'expiration_date'
    ]

    if field_name not in valid_fields:
        return jsonify({
            'error': f'Invalid field name. Valid fields: {valid_fields}'
        }), 400

    try:
        # Get document chunks to reconstruct text
        chunks = doc.chunks.order_by(DocumentChunk.chunk_index).all()
        document_text = "\n\n".join([chunk.text for chunk in chunks])

        if not document_text:
            return jsonify({'error': 'No text available for this document'}), 400

        # Extract single field
        from services.extractor import extract_field_api
        result = extract_field_api(field_name, document_text)

        # Update document if successful
        if result['status'] == 'success' and result['value']:
            new_value = result['value']
            if field_name == 'expiration_date':
                new_value = parse_date(new_value)
                if not new_value:
                    return jsonify({'error': 'Extracted expiration_date could not be parsed'}), 400
            setattr(doc, field_name, new_value)
            db.session.commit()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/re-extract/<int:doc_id>', methods=['POST'])
def re_extract_metadata(doc_id):
    """
    Re-extract all metadata fields for a document using parallel extraction.
    Useful if initial extraction failed or needs to be refreshed.

    POST /api/re-extract/<doc_id>

    Returns: Updated document metadata
    """
    doc = Document.query.get_or_404(doc_id)

    try:
        # Get document chunks to reconstruct pages
        chunks = doc.chunks.order_by(DocumentChunk.page_number, DocumentChunk.chunk_index).all()

        # Reconstruct pages from chunks
        pages_dict = {}
        for chunk in chunks:
            if chunk.page_number not in pages_dict:
                pages_dict[chunk.page_number] = []
            pages_dict[chunk.page_number].append(chunk.text)

        pages = [(page_num, '\n'.join(texts)) for page_num, texts in sorted(pages_dict.items())]
        document_text = '\n\n'.join([text for _, text in pages])

        if not document_text:
            return jsonify({'error': 'No text available for this document'}), 400

        # Prepare chunk data with stored embeddings
        chunk_texts = []
        chunk_embeddings = []
        for chunk in chunks:
            emb = chunk.get_embedding()
            if emb is None:
                continue
            chunk_texts.append(chunk.text)
            chunk_embeddings.append(emb)

        # Re-extract metadata with page tracking
        print(f"[RE-EXTRACTION] Starting for document {doc_id}")
        metadata = extract_metadata(
            document_text,
            pages_text=pages,
            chunk_texts=chunk_texts if chunk_texts else None,
            chunk_embeddings=chunk_embeddings if chunk_embeddings else None
        )

        summary = metadata.pop('_extraction_summary', {})
        page_refs = metadata.pop('_page_references', {})
        successful = summary.get('successful', 0)

        # Update document fields
        for key, value in metadata.items():
            if hasattr(doc, key):
                if key == 'expiration_date' and isinstance(value, str):
                    value = parse_date(value)
                setattr(doc, key, value)

        # Update page references
        for field_name, page_list in page_refs.items():
            pages_field = f"{field_name}_pages"
            if hasattr(doc, pages_field):
                pages_str = ','.join(str(p) for p in page_list) if page_list else None
                setattr(doc, pages_field, pages_str)

        # Update extraction status
        if successful >= 5:
            doc.extraction_status = 'success'
            doc.extraction_error = None
        else:
            doc.extraction_status = 'partial'
            doc.extraction_error = f"Only {successful}/10 fields extracted"

        db.session.commit()

        return jsonify({
            'message': 'Re-extraction completed',
            'extraction_summary': summary,
            'document': doc.to_dict()
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/update-metadata/<int:doc_id>', methods=['PUT'])
def update_metadata(doc_id):
    """
    Update document metadata fields manually.
    Allows users to edit extracted values.

    PUT /api/update-metadata/<doc_id>
    Body: {
        "irb_number": "new value",
        "principal_investigator": "new value",
        ...
    }

    Returns: Updated document
    """
    doc = Document.query.get_or_404(doc_id)
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Allowed fields to update
    allowed_fields = [
        'irb_number', 'principal_investigator', 'study_title',
        'study_purpose', 'inclusion_criteria', 'exclusion_criteria',
        'data_elements', 'funding_source', 'protocol_status', 'expiration_date'
    ]

    updated_fields = []

    try:
        for field, value in data.items():
            if field in allowed_fields and hasattr(doc, field):
                # Clean the value
                if value == '' or value == 'Not specified':
                    value = None
                elif isinstance(value, str):
                    value = value.strip() or None

                if field == 'expiration_date' and value:
                    parsed_date = parse_date(value)
                    if not parsed_date:
                        return jsonify({'error': 'Invalid expiration_date format. Use YYYY-MM-DD or a recognizable date.'}), 400
                    value = parsed_date

                setattr(doc, field, value)
                updated_fields.append(field)

        # Mark as manually edited
        doc.manually_edited = True
        doc.last_edited_date = datetime.utcnow()

        db.session.commit()

        return jsonify({
            'message': f'Updated {len(updated_fields)} fields',
            'updated_fields': updated_fields,
            'document': doc.to_dict()
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
