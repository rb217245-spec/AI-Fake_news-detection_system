# 🔎 TruthLens AI — Fake News Detector

TruthLens AI is an evidence-aware news verification and media literacy web application built with **Python**, **Streamlit**, the official **Google GenAI SDK**, and **Scikit-learn** Machine Learning.

It empowers readers to analyze news headlines, claims, and full articles against credible evidence patterns, identify emotional manipulation and linguistic red flags, and receive actionable verification steps.

---

## ✨ Features

- **Hybrid Analysis Architecture**:
  - **Google Gemini Generative AI**: In-depth claim extraction, plausibility reasoning, inconsistency detection, and structured verification guidance.
  - **Local Machine Learning Classifier**: Offline TF-IDF N-gram feature extraction and Logistic Regression classifier trained on verifiable news examples.
- **Robust Error Diagnostics**: Gracefully handles network timeouts, quota limits (429), authentication errors (401/403), and high-demand spikes (503) without crashing or silently falling back to fake results.
- **Enterprise-Grade Key Safety**: API keys are read strictly from `.env` or session memory, never exposed in logs, error traces, or UI elements, and automatically excluded from Git.
- **URL Article Extraction**: Built-in scraper using BeautifulSoup with strict timeout guards and private-network/SSRF blocking.
- **High-Performance Caching**: SHA-256 text hashing avoids redundant API calls and reduces quota consumption.
- **Educational Benchmarks**: Built-in pre-configured samples for quick demonstrations (e.g., Chandrayaan-3, T20 World Cup, Constitution of India, and viral hoaxes).

---

## 🚀 Quick Start on Windows

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system. Verify with:
```powershell
python --version
```

### 2. Open the Project Folder
Open PowerShell in the project directory:
```powershell
cd C:\Users\Lenovo\Downloads\fakenewsdetectionsystem
```

### 3. Create & Activate a Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Train the Local ML Classifier (Included)
The project comes with a pre-trained model, but you can retrain it anytime:
```powershell
python train_model.py
```

### 6. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in your text editor and add your API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```
> **Tip:** You can obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

### 7. Run the Application
```powershell
python -m streamlit run app.py
```
Streamlit will launch and automatically open `http://localhost:8501` in your default browser.

---

## 🛠️ The Original `AttributeError` & How It Was Resolved

### What caused the error?
Earlier versions produced the following failure:
```
Gemini analysis could not be completed (AttributeError). Using offline demo matching only.
```
**Root Causes Identified:**
1. **SDK Mixing / Incorrect Class Invocation**:
   The code imported the modern SDK (`from google import genai`), but attempted to instantiate a model using the legacy SDK syntax:
   `model = genai.GenerativeModel("gemini-3.5-flash-lite")`
   In the official `google-genai` SDK, `genai` has no attribute `GenerativeModel` (which belongs to legacy `google.generativeai`). This raised `AttributeError: module 'google.genai' has no attribute 'GenerativeModel'`.
2. **Unsupported Model ID**:
   Earlier iterations hardcoded deprecated model IDs (such as `gemini-2.5-flash` or `gemini-2.0-flash`), which returned `404 NOT_FOUND: model is no longer available to new users`.
3. **Silent Fallback Anti-Pattern**:
   The exception handler caught all errors and silently fell back to an offline 4-sample keyword matcher, misrepresenting failure as an offline demo match.

### How It Was Fixed:
- Standardized strictly on the official **Google GenAI SDK** (`from google import genai`, `from google.genai import types`).
- Initialized client cleanly with `client = genai.Client(api_key=key)`.
- Invoked model using the canonical SDK method: `client.models.generate_content(model=model_id, contents=prompt, config=...)`.
- Made the model configurable in `.env` (`GEMINI_MODEL=gemini-3.5-flash-lite`), with user selectable options in the sidebar.
- Added comprehensive diagnostic error cards displaying exact reasons (e.g., Quota, Invalid Key, Network Timeout) with a one-click Retry option, while keeping keys redacted.

---

## 📁 Project Structure

```
TruthLens_AI_Fake_News_Detector/
├── .streamlit/
│   └── config.toml          # Custom theme and server configuration
├── data/
│   └── news.csv             # Sample dataset with REAL and FAKE news examples
├── models/
│   └── fake_news_model.joblib # Trained TF-IDF + Logistic Regression pipeline
├── .env.example             # Safe template with placeholder values
├── .gitignore               # Comprehensive Git ignore rules (secrets protected)
├── app.py                   # Main Streamlit application entry point
├── ml_classifier.py         # Machine learning inference and diagnostic module
├── train_model.py           # ML training script with scikit-learn
├── requirements.txt         # Pinned production dependencies
└── README.md                # Documentation and setup guide
```

---

## 🔒 Safe GitHub Publishing Guide

To push this project to GitHub without leaking secrets:

1. **Verify `.gitignore`**:
   Ensure `.env` and `.streamlit/secrets.toml` are listed in `.gitignore`.
2. **Check Git Status Before Staging**:
   ```powershell
   git status
   ```
   Confirm that `.env` is **NOT** listed under untracked files.
3. **Initialize & Commit**:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: TruthLens AI Fake News Detector"
   ```
4. **Push to GitHub**:
   ```powershell
   git remote add origin https://github.com/your-username/TruthLens-Fake-News-Detector.git
   git branch -M main
   git push -u origin main
   ```

---

## ⚖️ Ethical & Educational Disclaimer
TruthLens AI is an advisory educational tool designed to assist in research and media literacy. Large language models and machine learning classifiers evaluate linguistic patterns and common knowledge; they cannot guarantee factual truth or replace thorough investigative journalism and verification with official primary sources.
