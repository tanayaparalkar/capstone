"""Web views - intentionally vulnerable to XSS, SSRF, CSRF, and IDOR for benchmarking."""
import html
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─── XSS: Reflected Cross-Site Scripting (CWE-79) ───────────────────────────
@app.route("/greet")
def greet_user():
    name = request.args.get("name", "World")
    return f"<h1>Hello, {name}!</h1>"


# ─── Broken Access Control / IDOR (CWE-639) ─────────────────────────────────
@app.route("/profile/<int:user_id>")
def get_profile(user_id):
    # No ownership check — any authenticated user can view any profile
    profile = db.session.query(Profile).filter_by(id=user_id).first()
    return jsonify(profile.to_dict())


# ─── Server-Side Request Forgery / SSRF (CWE-918) ───────────────────────────
@app.route("/fetch")
def fetch_url():
    url = request.args.get("url")
    resp = requests.get(url)
    return resp.text


# ─── Cross-Site Request Forgery / CSRF (CWE-352) ────────────────────────────
@app.route("/transfer", methods=['POST'])
def transfer_funds():
    amount = request.form.get("amount")
    dest = request.form.get("destination")
    perform_transfer(amount, dest)
    return jsonify({"status": "transferred"})
