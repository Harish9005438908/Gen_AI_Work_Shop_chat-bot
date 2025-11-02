# app.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai

# -----------------------------
# 1) HARDCODED API KEY
#    (rotate/change before sharing; don't commit to public repos)
# -----------------------------
GEMINI_API_KEY = "Your Api Key"
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    raise RuntimeError("Put your real Gemini API key in GEMINI_API_KEY.")

# Model (fast & affordable)
MODEL_NAME = "gemini-2.5-flash-lite"

genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# 2) PROMPT PROFILES (behavior presets)
#    Choose via request JSON: { "mode": "teacher" }
# -----------------------------
PROFILES = {
    "default": {
        "system": (
            "You are a concise, helpful assistant. Be accurate, avoid guessing, "
            "cite assumptions, and keep answers short unless asked for detail."
        ),
        "generation_config": {"temperature": 0.6, "top_p": 0.9, "top_k": 40, "max_output_tokens": 512},
    },
    "teacher": {
        "system": (
            "You are a friendly teacher for beginners. Explain step-by-step, use small examples, "
            "and avoid jargon. If code is requested, give a minimal runnable snippet."
        ),
        "generation_config": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_output_tokens": 700},
    },
    "support": {
        "system": (
            "You are an IT helpdesk agent for a corporate audience in India. Ask 1-2 clarifying "
            "questions before giving steps. Answers must be practical and bullet-pointed."
        ),
        "generation_config": {"temperature": 0.5, "top_p": 0.8, "top_k": 40, "max_output_tokens": 600},
    },
}

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history", [])             # [{role: "user"/"model", text: "..."}]
    mode = (data.get("mode") or "default").lower()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Pick profile (fallback to default)
    profile = PROFILES.get(mode, PROFILES["default"])
    system_instruction = profile["system"]
    generation_config = profile["generation_config"]

    # Build chat history for Gemini (keep last few turns)
    contents = []
    for turn in history[-8:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [turn.get("text", "")]})
    contents.append({"role": "user", "parts": [user_message]})

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )
        resp = model.generate_content(contents)
        text = resp.text or "(No response text)"
        return jsonify({"reply": text, "mode": mode})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # WARNING: debug=True is for local dev only
    app.run(host="0.0.0.0", port=8000, debug=True)