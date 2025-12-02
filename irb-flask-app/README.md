# IRB Flask App

Quick start to run the app locally.

1) Clone and enter the project  
`git clone <repo> && cd irb-flask-app`

2) Create and activate a virtual environment  
`python3 -m venv venv`  
`source venv/bin/activate`  (Windows: `venv\Scripts\activate`)

3) Install dependencies  
`pip install -r requirements.txt`

4) Set environment variables (at minimum)  
- `OPENAI_API_KEY`  
- Optional: `OPENAI_MODEL` (default `gpt-5-nano`), `EMBEDDING_MODEL` (default `text-embedding-3-large`).

5) Run the server  
`python app.py`  
The app starts on http://127.0.0.1:5000.

6) Use the UI  
- Upload a PDF/DOCX/TXT on the home page.  
- After processing, view metadata, edit it, or ask questions on the document page.  
- Search across documents via the search page.

Notes
- Uses SQLite by default; the database file is `irb_documents.db` in the project folder.  
- Upload limit defaults to 50MB.  
- Requires valid OpenAI credentials for both embeddings and chat calls.
