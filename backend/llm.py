"""Local LLM via Ollama - no API keys, 100% offline."""
import requests
from backend.config import settings

SYSTEM_PROMPT = (
    "You are a document assistant. Answer ONLY using the provided document excerpts.\n"
    "If the context does not contain the answer, say exactly: "
    "'The uploaded documents do not contain this information.'\n"
    "Never invent facts. Keep answers short and clear."
)


def build_context(sources: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(f"[Source {i} | {s['filename']} | part {s['chunk_index']}]\n{s['text']}")
    return "\n\n".join(parts)


def is_ollama_ready() -> bool:
    try:
        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def generate_answer(question: str, sources: list[dict]) -> str:
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context from documents:\n\n{build_context(sources)}\n\n"
                f"Question: {question}\n\nAnswer using only the context above:"
            )},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        r = requests.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        answer = r.json()["message"]["content"].strip()
        if not answer:
            raise RuntimeError("Empty answer from local LLM.")
        return answer
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve "
            "(and once: ollama pull qwen2.5:3b)"
        )
