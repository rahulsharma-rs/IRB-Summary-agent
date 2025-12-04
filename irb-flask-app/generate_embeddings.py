#!/usr/bin/env python3
"""
Migration script to generate document-level embeddings for existing documents.
Run this after adding the document_embedding column to the database.

Usage:
    python generate_embeddings.py
"""

import sys
from app import app, db
from models.database import Document
from services.searcher import create_document_search_profile
from services.embedder import embed_single


def generate_embeddings_for_all():
    """Generate document embeddings for all documents that don't have them."""

    with app.app_context():
        # Get all documents without embeddings
        docs_without_embeddings = Document.query.filter(
            db.or_(
                Document.document_embedding.is_(None),
                Document.document_embedding == ''
            )
        ).filter(
            Document.extraction_status.in_(['success', 'partial'])
        ).all()

        if not docs_without_embeddings:
            print("✅ All documents already have embeddings!")
            return

        print(f"Found {len(docs_without_embeddings)} documents without embeddings")
        print("Generating embeddings...\n")

        success_count = 0
        fail_count = 0

        for idx, doc in enumerate(docs_without_embeddings, 1):
            try:
                # Create search profile
                profile = create_document_search_profile(doc)

                if not profile.strip():
                    print(
                        f"⚠️  [{idx}/{len(docs_without_embeddings)}] Doc {doc.id} ({doc.filename}): No metadata to embed")
                    fail_count += 1
                    continue

                # Generate embedding
                embedding = embed_single(profile)

                # Save to database
                doc.set_document_embedding(embedding)
                db.session.commit()

                print(f"✅ [{idx}/{len(docs_without_embeddings)}] Doc {doc.id} ({doc.filename}): Embedding generated")
                success_count += 1

            except Exception as e:
                print(f"❌ [{idx}/{len(docs_without_embeddings)}] Doc {doc.id} ({doc.filename}): Error - {str(e)}")
                fail_count += 1
                db.session.rollback()
                continue

        print(f"\n{'=' * 60}")
        print(f"✅ Successfully generated: {success_count}")
        print(f"❌ Failed: {fail_count}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    print("🚀 Starting document embedding generation...\n")
    generate_embeddings_for_all()
    print("\n✨ Done!")