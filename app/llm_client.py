# app/llm_client.py
import os, json
import google.generativeai as genai

MODEL = os.getenv("AICOACH_MODEL", "gemini-2.5-flash-lite")
TEMP  = float(os.getenv("AICOACH_TEMPERATURE", "0.2"))

def gemini_reply(user_query: str, data: dict) -> str:
    """
    Call Gemini API to interpret AU payloads into a natural-language summary.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY not set."

    # configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    # Build prompt
    prompt = (
        "You are a concise coach reading facial Action Units (AUs) computed by OpenFace.\n"
        "Use ONLY the provided numbers; do not invent facts; do not give medical/clinical advice.\n"
        "Give 2–4 sentences. If trends look negative or arousal is low, add 1 practical tip.\n\n"
        f"User question: {user_query}\n\n"
        "Structured AU data (JSON):\n"
        f"{json.dumps(data, ensure_ascii=False, default=str)}\n\n"   # 👈 FIX HERE
        "Answer:"
    )

    # Call Gemini
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": TEMP}
    )

    return (resp.text or "").strip()
