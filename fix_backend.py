import os
import re

ROOT = r"C:\Users\VICTUS\Downloads\document-understanding-agent-main\document-understanding-agent-main"
RETRIEVER = os.path.join(ROOT, "backend", "retriever.py")
LLMPY = os.path.join(ROOT, "backend", "llm.py")
MAIN = os.path.join(ROOT, "backend", "main.py")


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ----------------------------
# 1. PATCH backend/retriever.py
# ----------------------------

def patch_retrieever():
    content = read_file(RETRIEVER)

    # 1a) Ensure import re
    if not re.match(r"^import re", content, flags=re.MULTILINE):
        content = "import re\n" + content

    # 1b) Add normalize_query / is_summary_query / is_followup_query if missing
    if "def normalize_query(" not in content:
        extra = '''

# ----------------------------
# QUERY NORMALIZATION
# ----------------------------

def normalize_query(query: str) -> str:
    query = (query or "").strip().lower()

    replacements = {
        "sumurizes": "summarizes",
        "suurizes": "summarizes",
        "sumurize": "summarize",
        "suurize": "summarize",
        "summurize": "summarize",
        "summurizes": "summarizes",
        "abt": "about",
        "pls": "please",
    }

    for old, new in replacements.items():
        query = re.sub(rf"\\b{re.escape(old)}\\b", new, query)

    return re.sub(r"\\s+", " ", query).strip()


def is_summary_query(query: str) -> bool:
    query = normalize_query(query)

    summary_words = (
        "summarize",
        "summary",
        "key points",
        "important points",
        "main points",
        "overview",
        "summarise",
        "briefly explain",
        "summarizes",
        "summarize it",
        "summarize the file",
        "summarize the document",
    )

    return any(word in query for word in summary_words)


def is_followup_query(query: str) -> bool:
    query = normalize_query(query)

    followup_phrases = (
        "more details",
        "more detail",
        "explain more",
        "tell me more",
        "continue",
        "elaborate",
        "what about that",
        "why",
        "how so",
        "about more",
        "more about",
    )

    return any(phrase in query for phrase in followup_phrases)

'''
        # Insert after imports, before first def/class
        lines = content.split("\n")
        insert_index = 0

        for i, line in enumerate(lines):
            if line.startswith("def ") or line.startswith("class "):
                insert_index = i
                break
            insert_index = i + 1

        new_lines = lines[:insert_index] + extra.split("\n") + lines[insert_index:]
        content = "\n".join(new_lines)

    # 1c) Fix retrieve_chunks to use normalization + summary/followup logic
    if "def retrieve_chunks(" in content:
        lines = content.split("\n")
        new_lines = []
        in_retrieve = False
        fixed_retrieve = False

        i = 0
        while i < len(lines):
            line = lines[i]

            if re.match(r"^\s*def retrieve_chunks\(", line):
                in_retrieve = True
                new_lines.append(line)
                i += 1
                continue

            if in_retrieve and not fixed_retrieve:
                # Detect main body: chunks = storage.get_chunks(...)
                if re.search(r"chunks\s*=\s*storage\.get_chunks\(", line):
                    # Determine indentation
                    m = re.match(r"^(?<ind>\s*)", line)
                    ind = m.group("ind") if m and "ind" in m.groupdict() else ""

                    # Add new logic
                    new_lines.append(f"{ind}# Normalize query")
                    new_lines.append(f"{ind}query = normalize_query(query)")
                    new_lines.append("")
                    new_lines.append(f"{ind}# Always use exact document filtering")
                    new_lines.append(f"{ind}chunks = storage.get_chunks(document_name=document_name)")
                    new_lines.append(f"{ind}if not chunks:")
                    new_lines.append(f"{ind}    return []")
                    new_lines.append("")
                    new_lines.append(f"{ind}# Summary requests need broad document context")
                    new_lines.append(f"{ind}if is_summary_query(query):")
                    new_lines.append(f"{ind}    return chunks[: min(len(chunks), 12)]")
                    new_lines.append("")
                    new_lines.append(f"{ind}# Vague follow-up requests need broader context")
                    new_lines.append(f"{ind}if is_followup_query(query):")
                    new_lines.append(f"{ind}    return chunks[: min(len(chunks), 8)]")
                    new_lines.append("")
                    new_lines.append(f"{ind}# Normal factual queries: use existing similarity search")
                    new_lines.append(f"{ind}results = existing_similarity_search(")
                    new_lines.append(f"{ind}    query=query,")
                    new_lines.append(f"{ind}    chunks=chunks,")
                    new_lines.append(f"{ind}    top_k=top_k,")
                    new_lines.append(f"{ind})")
                    new_lines.append("")
                    new_lines.append(f"{ind}return [")
                    new_lines.append(f"{ind}    r")
                    new_lines.append(f"{ind}    for r in results")
                    new_lines.append(f"{ind}    if r.get(\"score\", 0) >= 0.05")
                    new_lines.append(f"{ind}]")

                    # Skip old lines until next def/class or end of function
                    i += 1
                    while i < len(lines):
                        nxt = lines[i]
                        if re.match(r"^\s*def ", nxt) or re.match(r"^\s*class ", nxt):
                            i -= 1
                            break
                        i += 1

                    fixed_retrieve = True
                    in_retrieve = False
                    i += 1
                    continue

            new_lines.append(line)
            i += 1

        content = "\n".join(new_lines)

    write_file(RETRIEVER, content)
    print("backend/retriever.py patched.")


# ----------------------------
# 2. PATCH backend/llm.py
# ----------------------------

def patch_llm():
    content = read_file(LLMPY)

    # 2a) Add SYSTEM_PROMPT if not present
    if "SYSTEM_PROMPT =" not in content:
        prompt_block = '''SYSTEM_PROMPT = """
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

'''
        # Insert after imports
        lines = content.split("\n")
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_index = i + 1
            else:
                break

        new_lines = lines[:insert_index] + prompt_block.split("\n") + lines[insert_index:]
        content = "\n".join(new_lines)

    # 2b) Improve generate_answer to use conversation + better prompt
    if "def generate_answer(" in content:
        lines = content.split("\n")
        new_lines = []
        in_func = False
        fixed_func = False

        i = 0
        while i < len(lines):
            line = lines[i]

            if re.match(r"^\s*def generate_answer\(", line):
                in_func = True
                new_lines.append(line)
                i += 1
                continue

            if in_func and not fixed_func:
                # Look for prompt= or messages= construction
                if re.search(r"prompt\s*=", line) or re.search(r"messages\s*=", line):
                    m = re.match(r"^(?<ind>\s*)", line)
                    ind = m.group("ind") if m and "ind" in m.groupdict() else ""

                    new_lines.append(f"{ind}# Build conversation context")
                    new_lines.append(f"{ind}conversation_history = conversation_history or []")
                    new_lines.append(f"{ind}conversation_text = \"\"")
                    new_lines.append(f"{ind}if conversation_history:")
                    new_lines.append(f"{ind}    conversation_text = \"\\n\".join(")
                    new_lines.append(f"{ind}        f\"{{msg['role']}}: {{msg['content']}}\"")
                    new_lines.append(f"{ind}        for msg in conversation_history[-4:]")
                    new_lines.append(f"{ind}    )")
                    new_lines.append("")
                    new_lines.append(f"{ind}# Build context from chunks")
                    new_lines.append(f"{ind}context = \"\\n\\n\".join(chunk[\"text\"] for chunk in retrieved_chunks)")
                    new_lines.append("")
                    new_lines.append(f"{ind}prompt = f\"\"\"")
                    new_lines.append(f"{ind}Selected document: {{document_name}}")
                    new_lines.append("")
                    new_lines.append(f"{ind}Recent conversation:")
                    new_lines.append(f"{ind}{{conversation_text}}")
                    new_lines.append("")
                    new_lines.append(f"{ind}User request:")
                    new_lines.append(f"{ind}{{query}}")
                    new_lines.append("")
                    new_lines.append(f"{ind}Document context:")
                    new_lines.append(f"{ind}{{context}}")
                    new_lines.append("")
                    new_lines.append(f"{ind}Answer only from the selected document context.")
                    new_lines.append(f"{ind}\"\"\"")

                    # Skip old lines until next def/class
                    i += 1
                    while i < len(lines):
                        nxt = lines[i]
                        if re.match(r"^\s*def ", nxt) or re.match(r"^\s*class ", nxt):
                            i -= 1
                            break
                        i += 1

                    fixed_func = True
                    in_func = False
                    i += 1
                    continue

            new_lines.append(line)
            i += 1

        content = "\n".join(new_lines)

    write_file(LLMPY, content)
    print("backend/llm.py patched.")


# ----------------------------
# 3. PATCH backend/main.py
# ----------------------------

def patch_main():
    content = read_file(MAIN)

    # 3a) Add deduplicate_sources helper if not present
    if "def deduplicate_sources(" not in content:
        helper = '''

def deduplicate_sources(chunks):
    unique_sources = []
    seen_sources = set()

    for result in chunks:
        source = result.get("document_name") or result.get("filename")
        if source and source not in seen_sources:
            seen_sources.add(source)
            unique_sources.append(source)

    return unique_sources

'''
        # Insert before first @app. or def
        lines = content.split("\n")
        insert_index = len(lines)

        for i, line in enumerate(lines):
            if line.startswith("@app.") or (line.startswith("def ") and "(" in line):
                insert_index = i
                break

        new_lines = lines[:insert_index] + helper.split("\n") + lines[insert_index:]
        content = "\n".join(new_lines)

    # 3b) Fix /chat endpoint to use deduplicate_sources
    if '@app.post("/chat")' in content:
        lines = content.split("\n")
        new_lines = []
        in_chat = False
        fixed_chat = False

        i = 0
        while i < len(lines):
            line = lines[i]

            if '@app.post("/chat")' in line:
                in_chat = True
                new_lines.append(line)
                i += 1
                continue

            if in_chat and not fixed_chat:
                # Look for return ChatResponse( or return { ... answer
                if re.search(r"return.*ChatResponse\(", line) or (
                    re.search(r"return\s*\{", line) and "answer" in line
                ):
                    m = re.match(r"^(?<ind>\s*)", line)
                    ind = m.group("ind") if m and "ind" in m.groupdict() else ""

                    new_lines.append(f"{ind}# Deduplicate sources")
                    new_lines.append(f"{ind}sources = deduplicate_sources(chunks)")
                    new_lines.append("")
                    new_lines.append(f"{ind}# Build response")
                    new_lines.append(f"{ind}return ChatResponse(")
                    new_lines.append(f"{ind}    answer=answer,")
                    new_lines.append(f"{ind}    sources=sources,")
                    new_lines.append(f"{ind}    document_name=document_name,")
                    new_lines.append(f"{ind}    error=error,")
                    new_lines.append(f"{ind})")

                    # Skip old return line(s)
                    i += 1
                    while i < len(lines):
                        nxt = lines[i]
                        if re.match(r"^\s*def ", nxt) or re.match(r"^\s*@app\.", nxt):
                            i -= 1
                            break
                        i += 1

                    fixed_chat = True
                    in_chat = False
                    i += 1
                    continue

            new_lines.append(line)
            i += 1

        content = "\n".join(new_lines)

    write_file(MAIN, content)
    print("backend/main.py patched.")


# ----------------------------
# RUN ALL PATCHES
# ----------------------------

if __name__ == "__main__":
    patch_retrieever()
    patch_llm()
    patch_main()
    print("\n=== PATCH COMPLETE ===")
    print("Restart your backend now (stop + re-run uvicorn).")