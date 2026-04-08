import os, pickle, re, math, joblib
from urllib.parse import urlparse, urlsplit

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..")

# ── Load models ────────────────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "xgboost_phishing_model (1).pkl"), "rb") as f:
    phishing_model = pickle.load(f)

text_model = joblib.load(os.path.join(MODELS_DIR, "text_model.pkl"))

# ── URL feature extraction ─────────────────────────────────────────────────────
# Features match exactly what xgboost_phishing_model (1).pkl was trained on:
# url_length, num_dots, num_hyphens, num_underscores, num_slashes, num_digits,
# num_special_chars, has_at_symbol, has_double_slash, has_ip_address, has_https,
# num_subdomains, domain_length, tld_length, subdomain_length,
# has_suspicious_words, has_port, num_query_params, num_fragments, url_entropy,
# digit_ratio, letter_ratio, is_common_tld, path_length, has_redirect

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "account", "update", "secure", "banking",
    "confirm", "password", "credential", "paypal", "ebay", "amazon", "apple",
    "microsoft", "google", "facebook", "free", "lucky", "winner", "prize",
    "urgent", "suspended", "limited", "click", "here", "now", "paypal",
]
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "io", "co", "uk", "us", "ca",
    "de", "fr", "au", "in", "jp", "br", "it", "es", "ru", "cn",
}
IP_PATTERN = re.compile(
    r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])"
)

def _url_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())

def get_url_features(url: str) -> list:
    """Extract the 25 features expected by the XGBoost phishing model."""
    full_url = url if url.startswith("http") else "http://" + url
    parsed   = urlparse(full_url)
    hostname = parsed.hostname or ""
    path     = parsed.path or ""
    query    = parsed.query or ""
    fragment = parsed.fragment or ""

    # TLD & subdomain breakdown
    host_parts  = hostname.split(".")
    tld         = host_parts[-1] if host_parts else ""
    domain_part = host_parts[-2] if len(host_parts) >= 2 else hostname
    subdomain   = ".".join(host_parts[:-2]) if len(host_parts) > 2 else ""

    url_lower = url.lower()
    letters   = sum(c.isalpha() for c in url)
    digits    = sum(c.isdigit() for c in url)
    n         = len(url) or 1
    specials  = sum(not c.isalnum() and c not in "/:.-_?=#&@%+" for c in url)

    features = [
        len(url),                                          # url_length
        url.count("."),                                    # num_dots
        url.count("-"),                                    # num_hyphens
        url.count("_"),                                    # num_underscores
        url.count("/"),                                    # num_slashes
        digits,                                            # num_digits
        specials,                                          # num_special_chars
        1 if "@" in url else 0,                            # has_at_symbol
        1 if url.count("//") > 1 else 0,                  # has_double_slash
        1 if IP_PATTERN.search(hostname) else 0,           # has_ip_address
        1 if parsed.scheme == "https" else 0,              # has_https
        max(0, len(host_parts) - 2),                       # num_subdomains
        len(domain_part),                                  # domain_length
        len(tld),                                          # tld_length
        len(subdomain),                                    # subdomain_length
        1 if any(w in url_lower for w in SUSPICIOUS_WORDS) else 0,  # has_suspicious_words
        1 if (parsed.port and parsed.port not in (80, 443)) else 0, # has_port
        len(query.split("&")) if query else 0,             # num_query_params
        1 if fragment else 0,                              # num_fragments
        _url_entropy(url),                                 # url_entropy
        digits / n,                                        # digit_ratio
        letters / n,                                       # letter_ratio
        1 if tld.lower() in COMMON_TLDS else 0,            # is_common_tld
        len(path),                                         # path_length
        1 if url.count("//") > 1 or "redirect" in url_lower or "url=" in url_lower else 0,  # has_redirect
    ]
    return features


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict/phishing", methods=["POST"])
def predict_phishing():
    data = request.get_json(force=True)
    url  = data.get("text", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        feats = np.array([get_url_features(url)], dtype=float)
        pred  = int(phishing_model.predict(feats)[0])
        proba = phishing_model.predict_proba(feats)[0]

        label      = "Phishing" if pred == 1 else "Safe"
        confidence = float(proba[pred]) * 100

        return jsonify({
            "label":      label,
            "confidence": round(confidence, 2),
            "safe_score": round(float(proba[0]) * 100, 2),
            "risk_score": round(float(proba[1]) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/fakenews", methods=["POST"])
def predict_fakenews():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        pred  = int(text_model.predict([text])[0])
        proba = text_model.predict_proba([text])[0]

        # 0 = Fake, 1 = Real/True
        label      = "Real" if pred == 1 else "Fake"
        confidence = float(proba[pred]) * 100

        return jsonify({
            "label":      label,
            "confidence": round(confidence, 2),
            "fake_score": round(float(proba[0]) * 100, 2),
            "real_score": round(float(proba[1]) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
