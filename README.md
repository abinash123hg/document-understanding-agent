# Document Understanding Agent for Handwritten Text Recognition

A private, local document question-answering system whose primary capability is
reading **handwritten** pages. Upload a scanned or handwritten PDF, an image of
a handwritten note, or a normal digital document, then ask questions about that
one file. Answers are drawn only from the selected document, and every factual
sentence cites the excerpt it came from.

No API key appears anywhere in this repository. The only network access is the
one-time model download described below.

---

## The eight pipeline stages

Each stage is a separate module, so it can be tested and replaced on its own.

| # | Stage | Module |
|---|-------|--------|
| 1 | Image loading | `backend/imaging.py` |
| 2 | Preprocessing: denoise, binarize, deskew, line segmentation | `backend/imaging.py` |
| 3 | Handwriting recognition (TrOCR) | `backend/handwriting.py` |
| 4 | Text normalization | `backend/processor.py` |
| 5 | Chunking | `backend/processor.py` |
| 6 | Embedding and indexing | `backend/retriever.py` |
| 7 | Retrieval: dense + BM25 + rerank | `backend/retriever.py` |
| 8 | Answer generation and verification | `backend/llm.py` |

Routing is decided per page, not per file: a PDF with a real text layer is read
with PyMuPDF and never goes near the recogniser. Only pages with fewer than 20
embedded characters are rasterized and recognised, and those chunks are tagged
`content_type: "handwritten_ocr"`.

---

## Why TrOCR, and what was removed

- **Tesseract** was never installed or imported here. It is not in
  `requirements.txt`.
- **Docling** was removed. It is a general document converter, and its layout
  machinery is dead weight once per-page handwriting recognition is the actual
  requirement. It was also the heaviest dependency in the project.
- **TrOCR `microsoft/trocr-base-handwritten`** replaced it, because it is the
  only option here trained specifically on handwritten English lines rather
  than printed pages. OpenCV handles the geometry (deskew and line
  segmentation) so TrOCR receives one clean line at a time, which is what it
  expects.

Line boundaries are found on the binarized mask, but the crops themselves are
taken from the grayscale image: thresholding throws away the stroke weight and
grey levels the recogniser relies on.

---

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) running locally, with the answer model pulled
- Internet access **once**, on the first run, for the Hugging Face models
- About 4 GB free RAM while the recogniser and the embedding models are loaded

```bash
ollama pull qwen2.5:1.5b
```

The application never lets Ollama fetch a model on demand. If the configured
model is not already pulled, requests fail with a message naming the model and
the `ollama pull` command to run.

## Install and run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt

uvicorn backend.main:app --reload --port 8000
```

Then serve the UI from the repository root so the CORS defaults match:

```bash
python -m http.server 5500
```

and open `http://localhost:5500/frontend/index.html`.

To regenerate the handwritten sample the tests use:

```bash
python tools/make_handwritten_sample.py
```

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Model names and mode |
| `GET` | `/health` | Ollama reachability, whether the model is pulled, whether TrOCR is loaded, stored counts |
| `POST` | `/upload` | Index one file. Also available as `POST /documents/upload` |
| `GET` | `/documents` | List stored documents with page, chunk and content-type counts |
| `POST` | `/chat` | Ask a question about one named document |
| `DELETE` | `/documents/{name}` | Remove a file from both the chunk store and the vector index |

Accepted extensions: `.pdf .txt .md .docx .png .jpg .jpeg`. Maximum upload size
is 50 MB, enforced while streaming to disk, so an oversized file is rejected
before it has been fully read.

A `/chat` response always includes the evidence it used:

```json
{
  "answer": "Twelve villages were surveyed. [S1]",
  "document_name": "notes.pdf",
  "sources": [{
    "filename": "notes.pdf",
    "page_number": 3,
    "chunk_id": "3-1",
    "chunk_index": 7,
    "content_type": "handwritten_ocr",
    "extraction_method": "trocr",
    "score": 4.11,
    "text": "The survey covered twelve villages ..."
  }]
}
```

---

## Retrieval

One pipeline, in this order:

1. **Dense** search in ChromaDB, filtered to the selected document.
2. **BM25** over that document's chunks.
3. **Reciprocal Rank Fusion** of the two rankings, so agreement between them is
   rewarded without either score scale dominating.
4. **Cross-encoder rerank** (`ms-marco-MiniLM-L6-v2`), with a floor on the
   rerank score so weak matches are dropped rather than padded in.

Indexing happens once, at upload. Retrieval never re-indexes — that is what made
earlier versions slow and what allowed stale vectors to outlive a delete.

---

## How hallucination is blocked

Six independent measures, each covered by a test:

1. **Document isolation.** Retrieval is scoped to one document name and returns
   nothing when none is selected. There is no global fallback.
2. **Stale-vector purge.** Deleting or re-uploading clears both the JSON chunk
   store and the Chroma vectors, so removed text cannot be retrieved and cannot
   look grounded.
3. **Citation requirement.** Every factual sentence must carry an `[S<n]>`
   label. A draft with no label is discarded.
4. **Independent verifier.** A second pass checks each claim against the
   excerpts. It fails closed: any error in the verifier means no answer.
5. **OCR confidence gate.** If the only evidence is handwriting the recogniser
   scored below `HTR_MIN_CONFIDENCE`, the system refuses rather than answer
   from garbled text.
6. **Temperature 0.0**, plus a prompt-injection rule: excerpts are data, never
   commands.

When the evidence is insufficient the answer is exactly:

> I could not find enough information in the uploaded documents to answer this
> confidently.

---

## Tests

```bash
pytest                                    # 76 tests, about 20 seconds
pytest -m slow                            # +1 test: a real TrOCR pass, about 50s
pytest tests/test_retrieval.py -k leak    # the document-isolation guarantee
```

Five modules: `test_handwriting.py`, `test_processor.py`, `test_retrieval.py`,
`test_llm.py`, `test_api.py`.

Measured in this repository, Python 3.12.7, CPU only:

```
76 passed, 1 deselected        (pytest -m "not slow")
1 passed                       (pytest -m slow)
```

### Recognition accuracy

`sample_docs/handwritten_notes.png` is **synthetic** handwriting: five lines
rendered with the cursive *Ink Free* font and tilted 2.5 degrees, with the exact
text kept in `handwritten_notes_ground_truth.txt` so error rates are measurable.

Measured on that page, `microsoft/trocr-base-handwritten`, greedy decoding:

| Metric | Value |
|--------|-------|
| Lines segmented | 5 of 5 |
| Character error rate | 0.050 |
| Word error rate | 0.184 |
| Mean token confidence | 0.884 |

CER and WER describe the recogniser only. They are not retrieval or answer
accuracy, and **they are not a claim about real human handwriting** — a clean
synthetic font is an easier target than a person's notes. Treat these as a
pipeline check, not a benchmark.

### Speed

Same machine, CPU only:

| Operation | Time |
|-----------|------|
| Extract text from a 20-page digital PDF | 0.12 s |
| Full upload of that PDF (chunk + embed 120 chunks), steady state | ~1.2 s |
| First upload in a fresh process (loads the embedding models) | ~6 s |
| One handwritten PDF page through TrOCR | ~10-15 s |

A digital PDF never touches the recogniser, which is what keeps ordinary uploads
fast. Scanned pages cost what CPU inference costs, and `PDF_MAX_OCR_PAGES`
(default 50) makes a very large scan fail with an explanation instead of
appearing to hang.

---

## Configuration

See `.env.example`. Every value has a working default, so the application runs
with no `.env` file at all. Copy the example to `.env` to change anything.

---

## Project layout

```
backend/
  imaging.py       stages 1-2   OpenCV preprocessing and line segmentation
  handwriting.py   stage 3      TrOCR recognition and confidence
  processor.py     stages 4-5   extraction, normalization, chunking
  retriever.py     stages 6-7   Chroma + BM25 + RRF + rerank
  llm.py           stage 8      generation, citation and verification rules
  storage.py       persistence  JSON chunk store
  config.py                      settings
  main.py                        FastAPI routes
frontend/index.html              single-file UI
tools/make_handwritten_sample.py
tests/                           five modules, one per pipeline concern
sample_docs/                     digital and handwritten fixtures
```

---

## Known limits

- TrOCR is English-only and reads one line at a time. Writing that runs across
  columns, or on a slanted baseline, degrades it.
- Recognition runs on CPU in this configuration, so a many-page scan is slow. A
  GPU would change the timings, not the design.
- The bundled handwritten sample is synthetic. Add your own page to
  `sample_docs/` and re-measure before quoting any accuracy number.
- The verifier is the same local model as the generator. It catches restating
  errors well; it is not a proof of correctness.
- DOCX tables are flattened during extraction, so cell alignment can be lost
  even when every value survives.
