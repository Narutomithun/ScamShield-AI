import os, pickle, re, joblib
from urllib.parse import urlparse

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
with open(os.path.join(MODELS_DIR, "phishing_model.pkl"), "rb") as f:
    phishing_model = pickle.load(f)

text_model = joblib.load(os.path.join(MODELS_DIR, "text_model.pkl"))

with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

# ── URL feature extraction ─────────────────────────────────────────────────────

def has_ip(url):
    ip_pattern = re.compile(
        r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])"
    )
    return 1 if ip_pattern.search(url) else 0

def get_url_features(url: str) -> list:
    """Extract the 30 numerical features from a URL."""
    parsed = urlparse(url if url.startswith("http") else "http://" + url)
    hostname = parsed.hostname or ""
    path     = parsed.path or ""
    full     = url

    dots_in_host = hostname.count(".")
    subdomains   = dots_in_host - 1 if dots_in_host >= 2 else dots_in_host

    features = {
        "UsingIP":            has_ip(url),
        "LongURL":            1 if len(url) >= 75 else (0 if len(url) < 54 else 0),
        "ShortURL":           1 if any(s in url for s in ["bit.ly","goo.gl","tinyurl","ow.ly","t.co","tr.im","is.gd","cli.gs","yfrog.com","migre.me","ff.im","tiny.cc","url4.eu","twit.ac","su.pr","twurl.nl","snipurl.com","short.to","BudURL.com","ping.fm","post.ly","Just.as","bkite.com","snipr.com","flic.kr","loopt.us","doiop.com","short.ie","kl.am","wp.me","rubyurl.com","om.ly","to.ly","bit.do","t.co","lnkd.in","db.tt","qr.ae","adf.ly","goo.gl","bitly.com","cur.lv","tinyurl.com","ow.ly","bit.ly","ity.im","q.gs","is.gd","po.st","bc.vc","twitthis.com","u.to","j.mp","buzurl.com","cutt.us","u.bb","yourls.org","x.co","prettylinkpro.com","scrnch.me","filoops.info","vzturl.com","qr.net","1url.com","tweez.me","v.gd","tr.im","link.zip.net"]) else 0,
        "Symbol@":            1 if "@" in url else 0,
        "Redirecting//":      1 if url.count("//") > 1 else 0,
        "PrefixSuffix-":      1 if "-" in hostname else 0,
        "SubDomains":         0 if subdomains == 1 else (1 if subdomains == 2 else 1),
        "HTTPS":              1 if parsed.scheme == "https" else 0,
        "DomainRegLen":       0,   # requires WHOIS, default safe
        "Favicon":            0,
        "NonStdPort":         1 if parsed.port and parsed.port not in (80, 443) else 0,
        "HTTPSDomainURL":     1 if "https" in hostname else 0,
        "RequestURL":         0,
        "AnchorURL":          0,
        "LinksInScriptTags":  0,
        "ServerFormHandler":  0,
        "InfoEmail":          1 if "mailto:" in url else 0,
        "AbnormalURL":        0 if hostname and hostname in url else 1,
        "WebsiteForwarding":  0,
        "StatusBarCust":      0,
        "DisableRightClick":  0,
        "UsingPopupWindow":   0,
        "IframeRedirection":  0,
        "AgeofDomain":        0,
        "DNSRecording":       0,
        "WebsiteTraffic":     0,
        "PageRank":           0,
        "GoogleIndex":        1,
        "LinksPointingToPage":0,
        "StatsReport":        0,
    }

    return [features[col] for col in feature_columns]


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
