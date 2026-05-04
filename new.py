from flask import Flask, request, jsonify, render_template, session
from google import genai
from google.genai import types
import os

app = Flask(__name__)
app.secret_key = "AIzaSyCPX-upDAynjSDheMQyuMuvEHkCVqL_QlM"

# --- Load API Key safely ---
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not set in environment variables")

# --- Set up Gemini client ---
client = genai.Client(api_key=API_KEY)

# ✅ Use stable model
model = "gemini-2.5-flash"


# --- System Prompt ---
system_instruction = types.Content(
    role="user",
    parts=[
        types.Part.from_text(
            text=(
                "You are a certified fitness and health expert. "
                "Respond in a concise and precise manner by default and in short. "
                "Only provide detailed explanations if the user asks explicitly "
                "(e.g., says 'explain', 'in detail', or 'expand'). "
                "Focus on giving actionable health, workout, or nutrition advice. "
                "Be motivational and friendly, and avoid discussing diseases or medical conditions."
            )
        )
    ]
)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", chat=session.get("chat_history", []))


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    if "chat_history" not in session:
        session["chat_history"] = []

    request_history = [
        system_instruction,
        types.Content(role="user", parts=[types.Part(text=user_input)])
    ]

    try:
        # ✅ Use NON-streaming (more stable on deployment)
        response = client.models.generate_content(
            model=model,
            contents=request_history
        )

        full_response = response.text

        # Save chat
        session["chat_history"].append({"role": "user", "message": user_input})
        session["chat_history"].append({"role": "model", "message": full_response})
        session.modified = True

        return jsonify({"response": full_response})

    except Exception as e:
        print("🔥 ERROR:", str(e))  # shows in Render logs
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
