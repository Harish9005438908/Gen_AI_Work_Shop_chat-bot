import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.genai as genai
from google.genai import types

# -----------------------------
# 1) API KEY HANDLING
#    (Replace with your actual key or use os.getenv)
# -----------------------------
# WARNING: Hardcoding the key is not recommended for security! 
# You should use os.getenv("GEMINI_API_KEY") if possible.
GEMINI_API_KEY = "AIzaSyDTKpmRey3_tO72WLp6lZ4L9dzOckLpDgs" # <<< REPLACE THIS WITH YOUR REAL KEY

# if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyDTKpmRey3_tO72WLp6lZ4L9dzOckLpDgs":
#     raise RuntimeError("Put your real Gemini API key in GEMINI_API_KEY.")

# Model (fast & affordable)
MODEL_NAME = "gemini-2.5-flash-lite"

# Initialize the client. The key is passed directly to the constructor.
client = genai.Client(api_key=GEMINI_API_KEY)
GROUNDING_TOOL = types.Tool(google_search=types.GoogleSearch())

# -----------------------------
# 2) PROMPT PROFILES (behavior presets)
# -----------------------------
PROFILES = {
    "default": {
        "system": (
            "You are a concise, helpful assistant. Be accurate, avoid guessing, "
            "cite assumptions, and keep answers short unless asked for detail."
        ),
        "temperature": 0.6, 
        "top_p": 0.9, 
        "top_k": 40, 
        "max_output_tokens": 512
    },
    "teacher": {
        "system": (
            "You are a friendly teacher for beginners. Explain step-by-step, use small examples, "
            "and avoid jargon. If code is requested, give a minimal runnable snippet."
        ),
        "temperature": 0.7, 
        "top_p": 0.95, 
        "top_k": 40, 
        "max_output_tokens": 700
    },
    "support": {
        "system": (
            "You are an IT helpdesk agent for a corporate audience in India. Ask 1-2 clarifying "
            "questions before giving steps. Answers must be practical and bullet-pointed."
        ),
        "temperature": 0.5, 
        "top_p": 0.8, 
        "top_k": 40, 
        "max_output_tokens": 600
    },
}

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.get("/")
def index():
    # Note: You must have a 'templates/index.html' file for this to work
    return render_template("index.html")

@app.post("/chat")
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history", [])         # [{role: "user"/"model", text: "..."}]
    mode = (data.get("mode") or "default").lower()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Pick profile (fallback to default)
    profile = PROFILES.get(mode, PROFILES["default"])
    system_instruction = profile["system"]
    
    # 1. Extract individual parameters for generation_config
    generation_config = {
        "temperature": profile["temperature"],
        "top_p": profile["top_p"],
        "top_k": profile["top_k"],
        "max_output_tokens": profile["max_output_tokens"],
    }
    
    # Build chat history for Gemini (keep last few turns)
    contents = []
    for turn in history[-8:]:
        role = "user" if turn.get("role") == "user" else "model"
        # Ensure part is a dictionary with a 'text' key as expected by the new SDK
        contents.append({"role": role, "parts": [{"text": turn.get("text", "")}]}) 
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    # Initialize sources list
    sources = []
    
    try:
        # 2. Final Fix: Pass ALL configuration settings inside a single 'config' dictionary,
        #    INCLUDING the RAG tool.
        resp = client.models.generate_content(
            model=MODEL_NAME, 
            contents=contents,
            config={
                "system_instruction": system_instruction, # System instruction goes here
                "temperature": generation_config["temperature"],
                "top_p": generation_config["top_p"],
                "top_k": generation_config["top_k"],
                "max_output_tokens": generation_config["max_output_tokens"],
                "tools": [GROUNDING_TOOL] # <-- RAG/GROUNDING IS ADDED HERE
            }
        )
        
        text = resp.text or "(No response text)"
        
        # 3. Extract RAG/Grounding Sources (Citations)
        if resp.candidates and resp.candidates[0].grounding_metadata:
            metadata = resp.candidates[0].grounding_metadata
            if metadata.grounding_chunks:
                # The grounding chunks contain the URI (link) and title
                for chunk in metadata.grounding_chunks:
                    if chunk.web:
                        sources.append({
                            "title": chunk.web.title or "Source",
                            "uri": chunk.web.uri
                        })
        
        # 4. Update return value to include sources
        return jsonify({"reply": text, "mode": mode, "sources": sources})
    
    except Exception as e:
        error_message = str(e)
        if "API_KEY_INVALID" in error_message or "API key not valid" in error_message:
            return jsonify({"error": "Gemini API Key is invalid or rate-limited. Check your key and usage."}), 500
            
        return jsonify({"error": f"An unexpected error occurred: {error_message}"}), 500

if __name__ == "__main__":
    # WARNING: debug=True is for local dev only
    app.run(host="0.0.0.0", port=8000, debug=True)
