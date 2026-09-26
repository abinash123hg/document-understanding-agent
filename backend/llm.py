"""
Local LLM via Ollama – optimized for small models under 2GB
Strict document-only answering
"""

import requests
from backend.config import settings


SYSTEM_PROMPT = """
You are a private document question-answering assistant.

Rules:
1. Answer only using the supplied document context.
2. Never use outside knowledge.
3. If the user asks for a summary, summarize the supplied document context.
4. If the user asks for more details, expand on the previous topic using the supplied context.
5. If the user's message contains a typo, infer the intended request.
6. If the answer is not present in the context, say exactly:
   I could not find the answer in this document.
7. Do not invent facts.
8. Use clear formatting with headings or numbered points when useful.
"""

def build_context(sources: list[dict]) -> str:
    if not sources:
        return "No relevant information found."

    parts = []
    for i, s in enumerate(sources, 1):
        text = s.get("text", "").strip()
        if text:
            parts.append(f"[Excerpt {i}]\n{text}")
    return "\n\n".join(parts)


def is_ollama_ready() -> bool:
    try:
        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def generate_answer(question: str, sources: list[dict]) -> str:
    context = build_context(sources)

    user_message = f"""CONTEXT:
{context}

QUESTION:
{question}

Answer using only the CONTEXT above. If the answer is not in the CONTEXT, say exactly:
I could not find the answer in this document."""

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "stream": False,
      "options": {
    "temperature": 0.0,
    "top_p": 0.1,
    "repeat_penalty": 1.1,
    "num_predict": 250,        # shorter answers = much faster
    "num_ctx": 2048
}
    }

    try:
        r = requests.post(
            f"{settings.ollama_url}/api/chat",
            json=payload,
            timeout=120
        )
        r.raise_for_status()
        answer = r.json()["message"]["content"].strip()

        if not answer:
            return "I could not find the answer in this document."

        return answer

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")
    except Exception as e:
        raise RuntimeError(f"LLM error: {e}")