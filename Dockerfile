FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_BASEURLPATH=/irb

WORKDIR /app

# System deps for PyMuPDF/OpenCV/OCR pipeline
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr \
        qpdf \
        poppler-utils \
        ghostscript \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/.rag_cache /app/uploads && \
    chown -R appuser /app

USER appuser

EXPOSE 8502

CMD ["streamlit", "run", "irb_discover_v1.py"]