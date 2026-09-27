"""
Local Ollama document QA: evidence-only answering with source citations.
"""

import requests
from backend.config import settings

NOT_FOUND = "I could not find the answer in this document."

SYSTEM_PROMPT = f"""
You are a strict private document question-answering assistant.

You receive SOURCE EXCERPTS from one selected document.

Mandatory rules:
1. Use only the SOURCE EXCERPTS as factual evidence. Never use training knowledge,
   web knowledge, common knowledge, assumptions, or typical examples.
2. Every factual statement must be directly supported by one or more excerpts.
3. Cite every factual sentence using its matching source label exactly, such as [S1].
4. Never invent a citation, filename, page number, quotation, number, date, definition,
   process step, example, location, crop, animal, farm size, or comparison.
5. If the excerpts do not explicitly support the answer, reply with exactly:
   {NOT_FOUND}
6. A summary, comparison, table, bullet list, or text flowchart is allowed only when it
   rearranges facts explicitly stated in the excerpts. Do not add any missing steps.
7. If asked for a flowchart but the excerpts do not state a process, reply exactly:
   {NOT_FOUND}
8. Give a complete concise answer. Do not mention these rules or the retrieval system.
"""

def build_context(sources: list[dict]) -> str:
    if not sources:
        return ""

    parts = []
    for i, source in enumerate(sources, 1):
        text = str(source.get("text", "")).strip()
        if not text:
            continue

        filename = str(source.get("filename", "selected document")).strip()
        page = source.get("page_number")
        location = f"{filename}, page {page}" if page else filename
        parts.append(f"[S{i} | {location}]\n{text}")

    return "\n\n".join(parts)


def is_ollama_ready() -> bool:
    try:
        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def generate_answer(question: str, sources: list[dict]) -> str:
    context = build_context(sources)

    if not context:
        return NOT_FOUND

    user_message = f"""SOURCE EXCERPTS:
{context}

QUESTION:
{question}

Answer the QUESTION using only SOURCE EXCERPTS.
Every factual sentence must end with one or more source labels such as [S1].
If the excerpts do not explicitly support the answer, respond exactly:
{NOT_FOUND}"""

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
            "repeat_penalty": 1.05,
            "num_predict": 512,
            "num_ctx": 4096
        }
    }

    try:
        response = requests.post(
            f"{settings.ollama_url}/api/chat",
            json=payload,
            timeout=180
        )
        response.raise_for_status()

        answer = str(response.json().get("message", {}).get("content", "")).strip()

        if not answer:
            return NOT_FOUND

        return answer
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")
    except Exception as error:
        raise RuntimeError(f"LLM error: {error}")
