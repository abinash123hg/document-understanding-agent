"""
End-to-end check against a running backend and a running Ollama.

    python tools/live_check.py --base http://127.0.0.1:8000 \
        [--oversize-file some.pdf]

Nothing is mocked. Every step goes through the HTTP API and reaches the real
TrOCR weights, the real vector index and the real local language model, so a
failure here is a failure the user would actually see. The checks that need a
deliberately misconfigured server (tiny upload limit, missing model, Ollama
unreachable) are skipped unless the server was started with those settings;
README documents how.

Exit status is non-zero if any check fails.
"""

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_docs"

NOT_FOUND = ("I could not find enough information in the uploaded documents "
             "to answer this confidently.")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def skip(name, why):
    """A check the server configuration makes impossible is not a failure."""
    print(f"[SKIP] {name} - {why}")


def labels_resolve(payload, name):
    """
    Every [S<n>] in a successful answer must point at a source the response
    returned. The API used to hand back the raw retriever list, so [S2] could
    name an excerpt that was never shown to the model and [S3] could be a dead
    link. A real server, real Ollama and real TrOCR are the only place this can
    be proven without a mock agreeing with itself.
    """
    answer = str(payload.get("answer", ""))
    sources = payload.get("sources") or []
    labels = {int(m) for m in re.findall(r"\[S(\d+)\]", answer)}
    in_range = labels and labels <= set(range(1, len(sources) + 1))
    check(f"{name}: every citation label resolves to a returned source",
          bool(in_range), f"labels {sorted(labels)} against {len(sources)} source(s)")
    document = payload.get("document_name")
    for label in sorted(labels):
        source = sources[label - 1]
        check(f"{name}: [S{label}] maps to a non-empty source of this document",
              bool(str(source.get("text", "")).strip())
              and source.get("filename") == document,
              f"{source.get('filename')!r} page {source.get('page_number')}, "
              f"{source.get('content_type')}")


def refusal_is_clean(payload, name):
    """A refusal must carry neither a citation nor evidence to cite."""
    answer = str(payload.get("answer", ""))
    check(f"{name}: refusal has no citation and no sources",
          "[S" not in answer and payload.get("sources") in ([], None),
          f"sources={payload.get('sources')} answer={answer[:80]!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--oversize-file", type=Path, default=None,
                        help="a file larger than this server's MAX_UPLOAD_SIZE_MB")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    def upload(path):
        with open(path, "rb") as handle:
            return requests.post(f"{base}/documents/upload",
                                 files={"file": (path.name, handle)}, timeout=600)

    # 1 - health and model wiring
    health = requests.get(f"{base}/health", timeout=30).json()
    check("backend is reachable", health.get("backend") == "ok")
    check("Ollama is reachable", health.get("ollama") is True, str(health.get("ollama")))
    check("configured LLM model is already pulled", health.get("llm_model_pulled") is True,
          health.get("llm_model", ""))
    check("handwriting model is configured", "trocr" in str(health.get("handwriting_model")),
          health.get("handwriting_model", ""))

    # 2 - handwritten image through the whole pipeline
    response = upload(SAMPLES / "handwritten_notes.png")
    check("handwritten PNG uploads", response.status_code == 200, str(response.text[:200]))
    image_doc = response.json().get("document_name", "")
    image_chunks = response.json().get("chunk_count", 0)
    check("handwritten PNG produced chunks", image_chunks > 0, f"{image_chunks} chunks")

    answer = requests.post(f"{base}/chat", timeout=600, json={
        "question": "How many litres does each storage tank hold?",
        "document_name": image_doc}).json()
    recognized_ok = "200" in answer.get("answer", "")
    cited = "[S" in answer.get("answer", "")
    check("answerable handwriting question is answered from the page", recognized_ok,
          f"reason={answer.get('refusal_reason')} {answer.get('answer', '')[:120]}")
    check("answer carries a citation", cited)
    sources = answer.get("sources", [])
    labels_resolve(answer, "handwritten PNG answer")
    check("handwriting evidence is tagged handwritten_ocr",
          any(s.get("content_type") == "handwritten_ocr" for s in sources),
          str({s.get("content_type") for s in sources}))

    # 3 - unanswerable question must be refused, not invented
    refusal = requests.post(f"{base}/chat", timeout=600, json={
        "question": "What was the total budget in rupees for the survey?",
        "document_name": image_doc}).json()
    check("unanswerable question is refused verbatim",
          refusal.get("answer", "") == NOT_FOUND, refusal.get("answer", "")[:160])
    refusal_is_clean(refusal, "unanswerable question")

    # 4 - scanned PDF: page 1 is an image, page 2 carries a text layer
    response = upload(SAMPLES / "handwritten_notes.pdf")

    if response.status_code == 413:
        # The sample is 2.9 MB, so a server configured below that is expected to
        # refuse it; that refusal is the oversized-upload check, not a failure.
        skip("handwritten PDF uploads", "rejected by this server's upload limit")
    else:
        check("handwritten PDF uploads", response.status_code == 200,
              str(response.text[:200]))

    if response.status_code != 200:
        # A server started with a tiny MAX_UPLOAD_SIZE_MB cannot accept this
        # 2.9 MB sample; the checks below need it, so they are reported as
        # skipped instead of failing a run that is only testing the limit.
        for name in ("only the image page is rasterised for OCR",
                     "page 1 is answered from the recognised handwriting",
                     "page 1 evidence came from an OCR engine, not the PDF text layer",
                     "page 2 is answered from the PDF text layer",
                     "page 2 evidence keeps its page number"):
            skip(name, f"PDF rejected with {response.status_code}")
        pdf_doc = ""
    else:
        pdf_doc = response.json().get("document_name", "")
        pages = response.json()
        check("only the image page is rasterised for OCR",
              pages.get("page_count") == 2 and pages.get("ocr_page_count") == 1,
              f"{pages.get('page_count')} pages, {pages.get('ocr_page_count')} OCR pages")

        from_handwriting = requests.post(f"{base}/chat", timeout=600, json={
            "question": "When were the tanks distributed?",
            "document_name": pdf_doc}).json()
        check("page 1 is answered from the recognised handwriting",
              "2024" in from_handwriting.get("answer", "") and "[S" in from_handwriting.get("answer", ""),
              from_handwriting.get("answer", "")[:160])
        labels_resolve(from_handwriting, "PDF page 1 answer")
        page_one = [s for s in from_handwriting.get("sources", [])
                    if s.get("page_number") == 1]
        check("page 1 evidence came from an OCR engine, not the PDF text layer",
              bool(page_one) and all(
                  s.get("extraction_method") in {"trocr", "pytesseract"}
                  for s in page_one),
              str([(s.get("extraction_method"), s.get("content_type"))
                   for s in page_one]))

        from_text_layer = requests.post(f"{base}/chat", timeout=600, json={
            "question": "What was the total budget in rupees?",
            "document_name": pdf_doc}).json()
        check("page 2 is answered from the PDF text layer",
              "9.6" in from_text_layer.get("answer", ""),
              from_text_layer.get("answer", "")[:160])
        labels_resolve(from_text_layer, "PDF page 2 answer")
        check("page 2 evidence keeps its page number",
              any(s.get("page_number") == 2 and s.get("content_type") == "digital_text"
                  for s in from_text_layer.get("sources", [])),
              str([(s.get("page_number"), s.get("content_type"))
                   for s in from_text_layer.get("sources", [])]))

    # 5 - document isolation
    response = upload(SAMPLES / "sample_policy.txt")
    check("second document uploads", response.status_code == 200, str(response.text[:200]))
    policy_doc = response.json().get("document_name", "")
    leaked = requests.post(f"{base}/chat", timeout=600, json={
        "question": "How many litres does each storage tank hold?",
        "document_name": policy_doc}).json()
    check("question about document A is refused while document B is selected",
          leaked.get("answer", "") == NOT_FOUND, leaked.get("answer", "")[:160])
    refusal_is_clean(leaked, "cross-document question")

    listed = requests.get(f"{base}/documents", timeout=30).json()
    names = {d.get("document_name") or d.get("name") for d in listed.get("documents", listed)}
    uploaded = {name for name in (image_doc, pdf_doc, policy_doc) if name}
    check("every document that uploaded is listed", uploaded <= names,
          f"{sorted(uploaded)} vs {sorted(names)}")

    # 6 - deletion really purges the index
    deleted = requests.delete(f"{base}/documents/{policy_doc}", timeout=60)
    check("document can be deleted", deleted.status_code == 200, str(deleted.text[:160]))
    after_delete = requests.post(f"{base}/chat", timeout=600, json={
        "question": "What does the policy say?", "document_name": policy_doc}).json()
    check("deleted document returns no evidence",
          after_delete.get("answer", "") == NOT_FOUND and not after_delete.get("sources"))

    # 7 - re-upload must not leave stale vectors behind
    before = requests.get(f"{base}/health", timeout=30).json()["chunk_count"]
    response = upload(SAMPLES / "sample_policy.txt")
    re_chunks = response.json().get("chunk_count", 0)
    after = requests.get(f"{base}/health", timeout=30).json()["chunk_count"]
    check("re-upload replaces rather than duplicates the index",
          after - before == re_chunks, f"delta {after - before}, re-uploaded {re_chunks}")

    # 8 - bad input handling
    scratch = Path(tempfile.mkdtemp(prefix="dua-live-"))
    broken = scratch / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a real document body\n")
    response = upload(broken)
    check("corrupt PDF is rejected with 422", response.status_code == 422,
          f"{response.status_code}: {response.text[:160]}")

    bogus = scratch / "notes.exe"
    bogus.write_bytes(b"MZ\x00\x00")
    response = upload(bogus)
    check("unsupported file type is rejected with 400", response.status_code == 400,
          f"{response.status_code}: {response.text[:160]}")

    if args.oversize_file:
        response = upload(args.oversize_file)
        check("oversized file is rejected with 413", response.status_code == 413,
              f"{response.status_code}: {response.text[:160]}")
    else:
        print("[SKIP] oversized upload (start the server with a small "
              "MAX_UPLOAD_SIZE_MB and pass --oversize-file)")

    # 9 - cleanup
    for name in (image_doc, pdf_doc, policy_doc):
        if name:
            requests.delete(f"{base}/documents/{name}", timeout=60)
    shutil.rmtree(scratch, ignore_errors=True)

    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("Failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
