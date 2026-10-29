"""
TruthLens AI — Fake News Detector
Production-grade Streamlit application for evidence-aware news assessment.
Powered by Google GenAI SDK and local Machine Learning (TF-IDF + Logistic Regression).
"""

import os
import re
import json
import html
import hashlib
from typing import Dict, Any, Tuple, Optional
from urllib.parse import urlparse

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Official Google GenAI SDK
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Local Machine Learning Module
import ml_classifier

# Load local environment configuration
load_dotenv(override=False)

# Supported verified Gemini models
DEFAULT_MODEL = "gemini-3.5-flash-lite"
SUPPORTED_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
]

# Predefined educational benchmark samples for testing
EDUCATIONAL_SAMPLES = {
    "Chandrayaan-3 Landing (Credible)": (
        "India's Chandrayaan-3 mission successfully landed near the Moon's south polar region on "
        "23 August 2023. The mission was conducted by the Indian Space Research Organisation (ISRO), "
        "making India the fourth country to achieve a soft lunar landing."
    ),
    "ICC T20 World Cup 2024 (Credible)": (
        "India defeated South Africa by 7 runs in the final of the ICC Men's T20 World Cup on "
        "29 June 2024 in Barbados, securing their second T20 World Cup title under captain Rohit Sharma."
    ),
    "Constitution of India (Credible)": (
        "The Constitution of India came into effect on 26 January 1950, establishing India as a "
        "sovereign democratic republic. The date is commemorated annually as Republic Day."
    ),
    "Miracle Cure Hoax (Misleading Sample)": (
        "BREAKING: Doctors discover an all-natural herbal tea that completely eliminates 100% of "
        "viral infections and cancer cells in 24 hours. The medical establishment is trying to ban it "
        "to protect pharmaceutical profits! Drink 2 cups daily to guarantee immunity forever."
    ),
    "Free Money Phishing Claim (Misleading Sample)": (
        "URGENT GOVERNMENT NOTIFICATION: The Ministry of Finance has authorized an immediate direct cash "
        "subsidy of ₹1,00,000 for every smartphone owner. Forward this message to 10 friends and click "
        "the link below to register your bank account details before midnight or lose your grant."
    ),
}

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="TruthLens | AI News & Claim Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --brand-primary: #2563eb;
    --brand-dark: #0f172a;
    --brand-light: #f8fafc;
    --border-color: #e2e8f0;
}
.block-container {
    max-width: 1140px;
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
}
.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #1d4ed8 100%);
    color: #ffffff;
    padding: 32px 34px;
    border-radius: 18px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
}
.hero-kicker {
    text-transform: uppercase;
    letter-spacing: 2.2px;
    font-size: 11px;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 6px;
}
.hero-box h1 {
    color: #ffffff !important;
    font-size: 32px;
    font-weight: 800;
    margin: 4px 0 10px 0;
    letter-spacing: -0.5px;
}
.hero-box p {
    color: #e2e8f0 !important;
    font-size: 15px;
    line-height: 1.5;
    max-width: 780px;
    margin: 0;
}
.tag-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.24);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    color: #ffffff;
    margin-top: 16px;
}
.card {
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    margin-bottom: 16px;
}
.badge-credible {
    background-color: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 18px;
    display: inline-block;
}
.badge-misleading {
    background-color: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 18px;
    display: inline-block;
}
.badge-unverified {
    background-color: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 18px;
    display: inline-block;
}
.section-heading {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# --- Helper: Safe API Key Sanitizer ---
def sanitize_secret(text: str, secret: str) -> str:
    """Ensure the API key never appears in text or error messages."""
    if not secret or len(secret) < 8:
        return text
    # Mask the exact secret and any common API key shapes
    masked = text.replace(secret, "[REDACTED_API_KEY]")
    masked = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", masked)
    masked = re.sub(r"AQ\.[0-9A-Za-z-_]{40,}", "[REDACTED_API_KEY]", masked)
    return masked


# --- Helper: URL Extraction ---
def extract_article_from_url(url: str) -> Tuple[str, str]:
    """
    Safely extract article content from a public web page.
    Validates URL scheme, guards against SSRF, and enforces a strict timeout.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Please provide a valid URL starting with http:// or https://")

    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "::1"} or hostname.endswith(".local") or hostname.startswith("192.168.") or hostname.startswith("10."):
        raise ValueError("Access to local or private network addresses is prohibited.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    try:
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("The website took longer than 10 seconds to respond. Try pasting the article text directly.")
    except requests.exceptions.RequestException as err:
        raise ValueError(f"Could not reach the website: {err}")

    content_type = response.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        raise ValueError("The provided link does not appear to be an HTML webpage.")

    soup = BeautifulSoup(response.text, "html.parser")
    for elem in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
        elem.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else parsed.netloc

    # Extract article body
    article_elem = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in article_elem.find_all(["p", "h1", "h2", "h3"])]
    body_text = "\n\n".join(p for p in paragraphs if len(p) > 25)

    if len(body_text) < 80:
        body_text = article_elem.get_text(" ", strip=True)

    if len(body_text) < 80:
        raise ValueError("Could not extract sufficient text from this article. Please copy and paste the news text manually.")

    # Return trimmed text (up to 8,000 characters)
    return title, body_text[:8000].strip()


# --- Gemini Analysis Function ---
def analyze_with_gemini(news_text: str, api_key: str, model_id: str) -> Dict[str, Any]:
    """
    Perform evidence-aware news verification using Google GenAI SDK.
    Returns structured assessment without inventing citations.
    """
    client = genai.Client(api_key=api_key)

    # Trim input text to prevent excessive payload
    trimmed_text = news_text[:8000].strip()

    prompt = f"""You are TruthLens AI, an objective, evidence-aware news verification assistant.
Your task is to analyze the following news headline or article.

GUIDELINES:
1. Do not assert certainty without evidence. If a claim cannot be verified from reliable common knowledge or internal consistency, classify it as "Unverified".
2. Do not assume a surprising fact is false; evaluate credibility based on plausibility, language tone, internal consistency, and attribution.
3. NEVER invent or fabricate citations, URLs, quotes, or fake sources.
4. Distinguish between verified facts and preliminary assessments.

Return ONLY a valid JSON object matching this schema:
{{
  "verdict": "Likely Credible" | "Potentially Misleading" | "Unverified",
  "confidence": 85,
  "confidence_explanation": "Clear plain-language justification for why this confidence level was chosen",
  "summary": "Neutral, concise plain-language summary of the claim and assessment",
  "reasoning": [
    "Key analytical point 1",
    "Key analytical point 2"
  ],
  "red_flags": [
    "Potential sensationalism, missing author, urgency pressure, or internal inconsistency (empty list if none found)"
  ],
  "claims_to_verify": [
    "Specific factual assertion that should be cross-checked with primary sources"
  ],
  "verification_steps": [
    "Actionable step the user can take to independently verify the claim"
  ]
}}

NEWS TEXT TO ANALYZE:
{trimmed_text}
"""

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
    )

    raw_text = (response.text or "").strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError("Gemini returned a response that could not be parsed into valid JSON.")

    # Validate verdict normalization
    valid_verdicts = {"Likely Credible", "Potentially Misleading", "Unverified"}
    if data.get("verdict") not in valid_verdicts:
        # Normalize near matches
        v_lower = str(data.get("verdict", "")).lower()
        if "credible" in v_lower or "real" in v_lower or "true" in v_lower:
            data["verdict"] = "Likely Credible"
        elif "mislead" in v_lower or "fake" in v_lower or "false" in v_lower or "suspicious" in v_lower:
            data["verdict"] = "Potentially Misleading"
        else:
            data["verdict"] = "Unverified"

    # Enforce type guarantees
    for field in ("reasoning", "red_flags", "claims_to_verify", "verification_steps"):
        if not isinstance(data.get(field), list):
            data[field] = [str(data.get(field))] if data.get(field) else []

    if not isinstance(data.get("confidence"), (int, float)):
        data["confidence"] = 70.0

    data["model_used"] = model_id
    return data


# --- Diagnostic Message Generator ---
def format_gemini_error(exc: Exception, active_key: str, model_id: str) -> Dict[str, str]:
    """Produce safe, diagnostic error feedback for the user without exposing secrets."""
    err_str = str(exc)
    exc_type = type(exc).__name__

    title = "Gemini AI Analysis Error"
    detail = "An unexpected error occurred during API communication."
    suggestion = "Please check your configuration and try again."

    if isinstance(exc, genai_errors.ClientError) or "400" in err_str or "404" in err_str or "401" in err_str or "403" in err_str:
        if "401" in err_str or "403" in err_str or "API_KEY_INVALID" in err_str or "permission" in err_str.lower():
            title = "API Key Authentication Failed"
            detail = "The Gemini API key was rejected by Google AI Studio. It may be expired, mistyped, or lacks Gemini API permissions."
            suggestion = "Verify your key at https://aistudio.google.com/ and update GEMINI_API_KEY in your local .env file or sidebar."
        elif "404" in err_str or "NOT_FOUND" in err_str or "model" in err_str.lower():
            title = f"Model '{model_id}' Not Found or Inaccessible"
            detail = f"The model '{model_id}' is not recognized or not available to your API account."
            suggestion = "Select a verified model like 'gemini-3.5-flash-lite' or 'gemini-3.8-flash' in your .env or sidebar."
        elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            title = "Gemini API Quota Exceeded (429)"
            detail = "Your Google AI Studio API rate limit or project quota has been reached."
            suggestion = "Wait 60 seconds before retrying, or check quota limits in Google AI Studio."
        else:
            title = f"API Client Request Error ({exc_type})"
            detail = sanitize_secret(err_str, active_key)
            suggestion = "Check your prompt text and model parameters."

    elif isinstance(exc, genai_errors.ServerError) or "500" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
        title = "Google Gemini Service Temporarily Unavailable (503)"
        detail = "The Gemini model is experiencing a high-demand spike or temporary maintenance."
        suggestion = "Demand spikes usually resolve within a few seconds. Click the Retry button below."

    elif "timeout" in err_str.lower() or "connection" in err_str.lower():
        title = "Network Connection Timeout"
        detail = "The network connection to the Gemini API endpoint timed out."
        suggestion = "Check your internet connectivity and firewall settings, then retry."

    else:
        title = f"API Error: {exc_type}"
        detail = sanitize_secret(err_str[:250], active_key)
        suggestion = "Review the parameters and ensure dependencies are up to date."

    return {"title": title, "detail": detail, "suggestion": suggestion}


# --- UI Presentation: Display Results ---
def render_analysis_results(result: Dict[str, Any], ml_result: Optional[Dict[str, Any]] = None):
    st.markdown("<div class='section-heading'>Comprehensive Verification Report</div>", unsafe_allow_html=True)

    verdict = result.get("verdict", "Unverified")
    confidence = result.get("confidence", 70)
    conf_explanation = result.get("confidence_explanation", "Based on contextual analysis and available information.")
    model_name = result.get("model_used", "Gemini")

    # Top Metric Banner
    col1, col2 = st.columns([1.1, 2.2], gap="medium")

    with col1:
        st.markdown("<div class='card' style='text-align:center;'>", unsafe_allow_html=True)
        if verdict == "Likely Credible":
            st.markdown("<div class='badge-credible'>✓ Likely Credible</div>", unsafe_allow_html=True)
        elif verdict == "Potentially Misleading":
            st.markdown("<div class='badge-misleading'>⚠ Potentially Misleading</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='badge-unverified'>? Unverified Claim</div>", unsafe_allow_html=True)

        st.metric(label="Assessment Confidence", value=f"{confidence}%")
        st.caption(f"Confidence Rationale: {conf_explanation}")
        st.caption(f"Engine: {model_name}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-heading'>Executive Summary</div>", unsafe_allow_html=True)
        st.write(result.get("summary", "No summary provided."))
        st.caption("Notice: AI assessments are advisory research tools and do not constitute absolute proof.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Detailed Analysis Columns
    tab_gemini, tab_ml, tab_steps = st.tabs(["🧠 AI Reasoning & Warning Signs", "🤖 Local Machine Learning Model", "🧭 Independent Verification Guide"])

    with tab_gemini:
        col_reason, col_flags = st.columns(2, gap="medium")
        with col_reason:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 🧠 Primary Reasoning")
            reasons = result.get("reasoning", [])
            if reasons:
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.write("No specific reasoning breakdown available.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_flags:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 🚩 Linguistic & Factual Red Flags")
            flags = result.get("red_flags", [])
            if flags and any(f.strip() for f in flags):
                for f in flags:
                    st.markdown(f"- ⚠️ {f}")
            else:
                st.success("No obvious emotional manipulation, extreme sensationalism, or missing attribution flags detected.")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_ml:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🤖 Local ML Classifier (TF-IDF + Logistic Regression)")
        if ml_result and ml_result.get("available"):
            pred = ml_result.get("prediction", "N/A")
            ml_conf = ml_result.get("confidence", 0)
            raw = ml_result.get("raw_label", "")
            probs = ml_result.get("probabilities", {})

            if raw == "FAKE":
                st.error(f"**Local Classifier Prediction:** {pred}")
            else:
                st.success(f"**Local Classifier Prediction:** {pred}")

            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("Model Confidence", f"{ml_conf}%")
            with mcol2:
                if probs:
                    prob_text = " | ".join(f"{k}: {v}%" for k, v in probs.items())
                    st.caption(f"Class probabilities: {prob_text}")

            st.info(f"**Methodology:** {ml_result.get('method')}. {ml_result.get('disclaimer')}")
        elif ml_result and not ml_result.get("available"):
            st.warning(ml_result.get("error", "Local ML model not available."))
            st.caption("To train or retrain the local ML model, run `python train_model.py` in your terminal.")
        else:
            st.write("ML analysis was not run for this query.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_steps:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Claims Requiring Independent Verification")
        claims = result.get("claims_to_verify", [])
        if claims:
            for c in claims:
                st.markdown(f"- 🔎 **Key Claim:** {c}")
        else:
            st.write("No distinct sub-claims flagged for isolation.")

        st.divider()
        st.markdown("#### 🧭 Recommended Verification Actions")
        steps = result.get("verification_steps", [])
        if steps:
            for idx, s in enumerate(steps, start=1):
                st.markdown(f"**{idx}.** {s}")
        else:
            st.markdown("- Cross-check the claim across multiple reputable independent news publishers.")
            st.markdown("- Locate the original primary source (e.g., government gazette, scientific paper, court filing).")
        st.markdown("</div>", unsafe_allow_html=True)


# --- MAIN APPLICATION ENTRY POINT ---
def main():
    # Load configuration
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    env_model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.markdown("### ⚙️ TruthLens Settings")

        # Session API Key Override
        session_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            placeholder="Paste API key for this session",
            help="Your key is held in memory for this session and is never saved to public files or printed on screen.",
        ).strip()

        active_key = session_key or env_key

        if active_key:
            st.success("✓ API Key configured and active")
            st.caption("Key value is hidden for security.")
        else:
            st.warning("⚠️ No Gemini API key detected")
            st.caption("Add GEMINI_API_KEY to your local `.env` file or paste it above to enable live AI analysis.")

        st.divider()

        # Model Selector
        st.markdown("#### 🧠 Model Selection")
        selected_model = st.selectbox(
            "Active Gemini Model",
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(env_model) if env_model in SUPPORTED_MODELS else 0,
            help="Choose a verified model ID. Configurable in .env via GEMINI_MODEL."
        )

        st.divider()

        # ML Model Status
        st.markdown("#### 🤖 Local ML Status")
        ml_info = ml_classifier.get_model_info()
        if ml_info.get("available"):
            st.success(f"✓ Local ML Model Ready ({ml_info.get('file_size_kb')} KB)")
            st.caption("TF-IDF + Logistic Regression")
        else:
            st.info("Local ML Model Not Found")
            st.caption("Run `python train_model.py` to train local classifier.")

        st.divider()
        st.markdown("#### 🛡️ Privacy & Safety")
        st.caption("- Keys are read strictly from `.env` or session memory.")
        st.caption("- Real credentials are never committed to Git or exported in ZIPs.")
        st.caption("- AI outputs provide educational analysis, not legal or factual guarantees.")

    # --- Hero Header ---
    st.markdown("""
    <div class="hero-box">
        <div class="hero-kicker">AI-Powered Media Literacy & Fact-Checking Assistant</div>
        <h1>🔎 TruthLens Fake News Detector</h1>
        <p>Analyze news headlines, claims, and articles using Google Gemini generative AI paired with local Machine Learning NLP classifiers. Discover warning signs, verify claims, and protect against misinformation.</p>
        <span class="tag-pill">Google GenAI SDK · TF-IDF NLP · Safe URL Scraper · Zero Secret Leakage</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Input Section ---
    st.markdown("<div class='section-heading'>Submit News for Verification</div>", unsafe_allow_html=True)

    tab_text, tab_url = st.tabs(["✍️ Paste News Text", "🔗 Fetch from Public Article URL"])

    input_text = ""
    submit_trigger = False

    with tab_text:
        col_sample, col_clear = st.columns([3, 1])
        with col_sample:
            sample_choice = st.selectbox(
                "Or load a pre-configured educational sample:",
                options=["Select a benchmark sample to test…", *EDUCATIONAL_SAMPLES.keys()],
                index=0,
            )
        with col_clear:
            st.write("")
            if st.button("🧹 Reset Input", use_container_width=True):
                st.session_state["news_input_area"] = ""
                st.session_state.pop("current_analysis", None)
                st.session_state.pop("error_info", None)
                st.rerun()

        # Handle sample selection
        default_val = ""
        if sample_choice != "Select a benchmark sample to test…":
            default_val = EDUCATIONAL_SAMPLES[sample_choice]

        input_text = st.text_area(
            "News headline, claim, or full article text:",
            value=default_val,
            height=180,
            placeholder="Paste news text here (minimum 15 characters)…",
            key="news_input_area",
        )

        submit_text = st.button("🔎 Analyze News Claim", type="primary", use_container_width=True, key="btn_submit_text")
        if submit_text:
            submit_trigger = True

    with tab_url:
        st.caption("Extract and analyze text directly from a public news article URL.")
        url_input = st.text_input("Public Article URL:", placeholder="https://example-news.com/article-slug")
        submit_url = st.button("🌐 Fetch and Analyze URL", type="primary", use_container_width=True, key="btn_submit_url")

        if submit_url:
            if not url_input.strip():
                st.error("Please provide an article URL.")
            else:
                with st.spinner("Fetching and parsing article content..."):
                    try:
                        extracted_title, extracted_body = extract_article_from_url(url_input.strip())
                        st.info(f"**Article Extracted:** {extracted_title}")
                        input_text = f"Title: {extracted_title}\n\n{extracted_body}"
                        submit_trigger = True
                    except Exception as exc:
                        st.error(f"URL Extraction Failed: {str(exc)}")

    # --- Execution Logic ---
    if submit_trigger:
        cleaned_query = (input_text or "").strip()

        if len(cleaned_query) < 15:
            st.error("Please provide a little more text to analyze (at least 15 characters).")
        else:
            # Check caching: Avoid re-querying identical text unless forced
            query_hash = hashlib.sha256(cleaned_query.encode("utf-8")).hexdigest()
            cached_data = st.session_state.get("cached_analysis", {})

            # 1. Run Local Machine Learning Classifier
            ml_prediction = ml_classifier.predict_text(cleaned_query)

            # 2. Run Gemini AI Verification
            if not active_key:
                st.warning("⚠️ No Gemini API Key configured. Showing local Machine Learning analysis only.")
                # Show safe ML-only fallback with clear message
                st.session_state["current_analysis"] = {
                    "verdict": "Unverified",
                    "confidence": 50,
                    "confidence_explanation": "Gemini AI key is missing. Live LLM reasoning is paused.",
                    "summary": "No Gemini API key is configured. The local ML classifier evaluated lexical patterns below, but factual verification requires an active Gemini key.",
                    "reasoning": [
                        "Gemini API key is required for multi-step reasoning, claim extraction, and evidence assessment.",
                        "Add GEMINI_API_KEY to your .env file or the sidebar password input to unlock full AI verification."
                    ],
                    "red_flags": ["Live AI fact-checking is currently inactive."],
                    "claims_to_verify": ["All claims remain unverified by AI."],
                    "verification_steps": [
                        "Obtain a free API key from Google AI Studio: https://aistudio.google.com/",
                        "Add GEMINI_API_KEY to .env and refresh."
                    ],
                    "model_used": "Local Mode Only",
                }
                st.session_state["ml_result"] = ml_prediction
                st.session_state.pop("error_info", None)

            elif query_hash in cached_data and cached_data[query_hash].get("model") == selected_model:
                # Use cached analysis
                st.session_state["current_analysis"] = cached_data[query_hash]["data"]
                st.session_state["ml_result"] = ml_prediction
                st.session_state.pop("error_info", None)

            else:
                with st.spinner("Analyzing news claims with Gemini AI..."):
                    try:
                        ai_assessment = analyze_with_gemini(cleaned_query, active_key, selected_model)
                        st.session_state["current_analysis"] = ai_assessment
                        st.session_state["ml_result"] = ml_prediction
                        st.session_state.pop("error_info", None)

                        # Update cache
                        if "cached_analysis" not in st.session_state:
                            st.session_state["cached_analysis"] = {}
                        st.session_state["cached_analysis"][query_hash] = {
                            "data": ai_assessment,
                            "model": selected_model
                        }
                    except Exception as exc:
                        # Capture structured diagnostic error without leaking key
                        diag = format_gemini_error(exc, active_key, selected_model)
                        st.session_state["error_info"] = diag
                        st.session_state["ml_result"] = ml_prediction
                        st.session_state.pop("current_analysis", None)

    # --- Render Error Card (If Error Occurred) ---
    if st.session_state.get("error_info"):
        err = st.session_state["error_info"]
        st.divider()
        st.error(f"### ⚠️ {err['title']}")
        st.markdown(f"**Diagnostic Details:** {err['detail']}")
        st.markdown(f"💡 **Suggested Action:** {err['suggestion']}")

        col_retry, col_space = st.columns([1, 3])
        with col_retry:
            if st.button("🔄 Retry Analysis", use_container_width=True):
                st.session_state.pop("error_info", None)
                st.rerun()

        # Still show ML result if available
        if st.session_state.get("ml_result"):
            st.divider()
            st.markdown("### 🤖 Local Machine Learning Analysis (Offline Backup)")
            render_analysis_results(
                {
                    "verdict": "Unverified",
                    "confidence": 40,
                    "confidence_explanation": "Gemini analysis encountered an error. Showing local model evaluation only.",
                    "summary": f"Gemini API was temporarily unreachable due to: {err['title']}. Local ML analysis evaluated text patterns.",
                    "reasoning": [f"API Error: {err['detail']}", "Independent fact-checking should be conducted."],
                    "red_flags": [],
                    "claims_to_verify": [],
                    "verification_steps": ["Verify through reputable news outlets."],
                    "model_used": "Offline Backup",
                },
                ml_result=st.session_state["ml_result"]
            )

    # --- Render Successful Results ---
    elif st.session_state.get("current_analysis"):
        st.divider()
        render_analysis_results(
            st.session_state["current_analysis"],
            ml_result=st.session_state.get("ml_result")
        )

    # --- Footer ---
    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:13px;'>"
        "TruthLens AI Fake News Detector · Powered by Google GenAI SDK & Scikit-learn · "
        "Advisory tool designed to augment media literacy, not replace independent research."
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
