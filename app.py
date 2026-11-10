from flask import Flask, jsonify, render_template, request
from engine import CoverageEngine

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16_384
engine = CoverageEngine()

@app.get("/")
def index():
    return render_template("index.html", mode=engine.mode)

@app.post("/api/ask")
def ask():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a question and role."), 400
    question, role = payload.get("question"), payload.get("role", "patient")
    if not isinstance(question, str) or not 2 <= len(question.strip()) <= 600:
        return jsonify(error="Enter a question between 2 and 600 characters."), 400
    if role not in ("patient", "nurse"):
        return jsonify(error="Choose Patient or Nurse."), 400
    return jsonify(engine.ask(question.strip(), role))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
