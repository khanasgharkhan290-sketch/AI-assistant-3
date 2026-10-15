# AI-assistant-3
# 📄 Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get an **estimated ATS score**, a category breakdown,
keyword analysis, prioritized improvements and suggested bullet rewrites — powered by
**Google Gemini Flash** and built with **Streamlit**.

> ⚠️ The score is an AI-based estimate, not the output of a real ATS. Use it as guidance.

## Features
- PDF / DOCX / TXT parsing (including DOCX tables)
- Optional job description for accurate keyword matching
- Weighted score across 5 categories (keywords, content, impact, formatting, structure)
- Prioritized improvements, matched/missing keywords, bullet rewrites
- Shows the exact text an ATS would extract from your file
- Downloadable Markdown report

## Run locally
```bash
git clone https://github.com/<your-username>/resume-ats-checker.git
cd resume-ats-checker
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/apikey>, then either:

- paste it into the app sidebar, **or**
- create `.streamlit/secrets.toml` containing:
  ```toml
  GEMINI_API_KEY = "your-key-here"
  ```
- or set an environment variable `GEMINI_API_KEY`.

Start the app:
```bash
streamlit run app.py
```

## Configuration
| Setting | How | Default |
|---|---|---|
| API key | sidebar, `st.secrets["GEMINI_API_KEY"]`, or env var | – |
| Model | sidebar field or `GEMINI_MODEL` env var | `gemini-2.5-flash` |

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public or private).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → pick the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Project structure
```
app.py            # Streamlit app + Gemini logic
requirements.txt  # Dependencies
README.md
```

## Limitations
- Scanned/image-only resumes can't be read (an ATS can't read them either).
- Resumes are truncated to ~20,000 characters.
- Never commit your API key. `.gitignore` already excludes `secrets.toml` and `.env`.
