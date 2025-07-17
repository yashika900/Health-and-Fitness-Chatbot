from flask import Flask, request, jsonify, render_template, session
from google import genai
from google.genai import types

app = Flask(__name__)
app.secret_key = "your_secret_key"  # For session storage

# --- Set up Gemini API client ---
client = genai.Client(
    api_key="AIzaSyD4APxUZUhzlz_Tdw697wZUqVPr1Y6ojjs"
)

model = "gemini-2.5-pro"

generate_content_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=-1)
)

# --- System Prompt: Fitness Expert Role with Precision ---
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

    # Prepare the input for model (system + latest user prompt only)
    request_history = [
        system_instruction,
        types.Content(role="user", parts=[types.Part(text=user_input)])
    ]

    try:
        full_response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=request_history,
            config=generate_content_config
        ):
            if chunk.text:
                full_response += chunk.text

        # Save full interaction in session history
        session["chat_history"].append({"role": "user", "message": user_input})
        session["chat_history"].append({"role": "model", "message": full_response})
        session.modified = True

        return jsonify({"response": full_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)