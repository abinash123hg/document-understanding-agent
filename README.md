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
- **TrOCR `microsoft/trocr-base-handwritten`** remains the default. On the
  repository's synthetic sample, its CER and WER matched the small model; this
  larger model had no measurable disadvantage on that test. This does not
  establish how either model performs on real handwriting. Set
  `HTR_MODEL=microsoft/trocr-small-handwritten` as a CPU speed option; its real-
  handwriting accuracy has not been measured here. OpenCV handles geometry
  (deskew and line segmentation) so TrOCR receives one clean line at a time.

grey levels the recogniser relies on. Each line is then trimmed horizontally to
Line boundaries are found on the binarized mask, but the crops themselves are
taken from the grayscale image: thresholding throws away the stroke weight and
grey levels the recogniser relies on. Each line is then trimmed horizontally to
its own ink, avoiding unnecessary blank space around the text.

---

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) running locally, with the answer model pulled
- Internet access **once**, on the first run, for the Hugging Face models

```bash
ollama pull qwen2.5:1.5b
```

The application never lets Ollama fetch a model on demand. If the configured
model is not already pulled, requests fail with a message naming the model and
the `ollama pull` command to run.

## Install and run

### Windows: one click

Double-click `start.bat`. It checks the virtual environment and the installed
packages first — anything already present is reused, anything missing is
downloaded once — then opens two terminal windows (backend on
`http://127.0.0.1:8000`, frontend on `http://localhost:5500`), waits for the
backend to answer, and opens Microsoft Edge on the app. It also warns if
Ollama is not running. Closing both windows stops the app.

### Any platform: manual

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt

uvicorn backend.main:app --reload --port 8000
```

In a second terminal, serve the UI — only the `frontend/` folder is exposed:

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

and open `http://localhost:5500`.

To regenerate the handwritten sample the tests use, then re-measure it:

```bash
python tools/make_handwritten_sample.py
python tools/evaluate_handwriting.py
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

A `/chat` response contains `answer`, `document_name`, and its retrieved
`sources`. A refusal also includes `refusal_reason`: `no_evidence`,
`low_handwriting_confidence`, `untraceable_citation`, `unsupported_claim`, or
`not_answerable`, so a refusal says which guard stopped the answer rather than
leaving the reading of it to guesswork.

---

## Retrieval

One pipeline, in this order:

1. **Dense** search in ChromaDB, filtered to the selected document.
2. **BM25** over that document's chunks.
3. **Reciprocal Rank Fusion** of the two rankings, so agreement between them is
   rewarded without either score scale dominating.
4. **Cross-encoder rerank** (`ms-marco-MiniLM-L6-v2`). Candidates are kept
   relative to the best score for that query (`RERANK_MARGIN`) instead of
   against an absolute cutoff: the score an excerpt receives depends on how
   well-formed the passage is, so a fixed floor silently discards every
   candidate from a chunked document (measured on a real portfolio PDF: the
   best score obtainable for an answer-bearing query was -10.4) while still
   letting weak matches through on a tidy one. The ranking separates on-topic
   from off-topic; the answer stage decides whether to reply.

Indexing happens once, at upload. Retrieval never re-indexes — that is what made
earlier versions slow and what allowed stale vectors to outlive a delete.

---

## How hallucination is blocked

Seven independent measures, each covered by a test:

1. **Document isolation.** Retrieval is scoped to one document name and returns
   nothing when none is selected. There is no global fallback.
2. **Stale-vector purge.** Deleting or re-uploading clears both the JSON chunk
   store and the Chroma vectors, so removed text cannot be retrieved and cannot
   look grounded.
3. **Citation requirement.** Every factual sentence must carry an `[S<n]>`
   label. A draft with no label gets one retry that shows the model its own
   answer and asks for the labels; a second miss becomes the refusal. The one
   exception is an answer over a single excerpt, where which evidence the fact
   came from is already settled and only the label was missing — the label is
   then appended, and the verifier still has to approve every claim.
4. **Citations must resolve.** A label pointing outside the supplied excerpts
   (`[S4]` when two excerpts exist) is a fabricated reference, so the answer is
   refused instead of being shown with a dead link. Outside the single-excerpt
   case above, the system never writes a label on the model's behalf.
5. **Independent verifier.** A second pass checks that each claim is stated in
   the excerpts *and* that the draft answers the question asked. It fails
   closed: any error in the verifier means no answer.
6. **OCR confidence gate.** An excerpt is only shown to the model, and can only
   be cited, when it is digital text or handwriting at or above
   `HTR_MIN_CONFIDENCE`. Garbled handwriting is removed from the evidence
   rather than merely flagged, so a low-confidence line cannot be quoted as an
   answer; if nothing trustworthy remains, the system refuses.
7. **Temperature 0.0**, plus a prompt-injection rule: excerpts are data, never
   commands.

When the evidence is insufficient the answer is exactly:

> I could not find enough information in the uploaded documents to answer this
> confidently.

---

## Tests

```bash
pytest
pytest -m slow
pytest tests/test_retrieval.py -k leak
```

Five modules: `test_handwriting.py`, `test_processor.py`, `test_retrieval.py`,
`test_llm.py`, `test_api.py`. The one skip is
`test_recognition_on_a_real_handwritten_sample`, which stays skipped until a real
page is added (see below).

Measured in this run:

```
96 passed, 1 skipped           (pytest -q)
1 passed, 1 skipped, 95 deselected (pytest -m slow -q)
1 passed, 12 deselected       (pytest tests/test_retrieval.py -k leak -q)
```

`tools/live_check.py` runs the same guarantees against a **real** running
backend, Ollama and TrOCR rather than mocks - upload, recognition, cited answer,
refusal, isolation, delete, re-upload, corrupt file, invalid file:

```bash
uvicorn backend.main:app --port 8000
python tools/live_check.py --base http://127.0.0.1:8000

# oversize rejection needs a server whose limit the sample can exceed
MAX_UPLOAD_SIZE_MB=1 uvicorn backend.main:app --port 8001
python tools/live_check.py --base http://127.0.0.1:8001 --oversize-file big.pdf
```

### Recognition accuracy

`sample_docs/handwritten_notes.png` is **synthetic** handwriting: five lines
rendered with the cursive *Ink Free* font, with the exact
text kept in `handwritten_notes_ground_truth.txt` so error rates are measurable.
The generator sizes the script from the measured text and rotates with
`expand=True`, so no line can be cut off by the page edge.

Measured on that synthetic page in this run with base TrOCR, greedy decoding:

| Lines segmented | Character error rate | Word error rate | Mean token confidence |
|-----------------|----------------------|-----------------|-----------------------|
| 5 of 5 | 0.017 | 0.204 | 0.993 |

```bash
python tools/evaluate_handwriting.py
```

CER and WER are computed over text lowercased and with whitespace collapsed, and
spaces are deliberately kept in the character stream so CER cannot report a
flattering zero.

The word error rate is affected by sentence-final punctuation being emitted as
its own token; the synthetic ground truth and recognised output can therefore
tokenize punctuation differently.

**Real human handwriting has not been measured.** These synthetic results do
not predict accuracy on a person's writing. Before trusting any score, measure
your own handwritten page and transcription with `tools/evaluate_handwriting.py`
(see `sample_docs/real_handwritten/README.md`):

```bash
python tools/evaluate_handwriting.py \
  --image sample_docs/real_handwritten/page1.png \
  --ground-truth sample_docs/real_handwritten/ground_truth.txt \
  --label "Real handwritten sample"

pytest -m slow
```

### Speed

Same machine, CPU only:

| Operation | Time |
|-----------|------|
| Extract generated 20-page digital PDF | 0.187 s |
| Warm full upload of generated PDF (160 chunks; chunk + embed) | 0.954 s |
| Full upload in a fresh process (model load + indexing) | 7.562 s |
| Warm OCR of the 5-line synthetic image with base TrOCR | 8.766 s |

The digital timing fixture is generated and contains no OCR pages. The cold
measurement includes backend imports and loading the cached embedding model;
the subprocess wall time was 12.952 s. All handwriting accuracy values above
are synthetic; accuracy on real handwriting remains unmeasured.

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
start.bat                        Windows one-click launcher
tools/
  make_handwritten_sample.py     regenerate the synthetic page, PDF and ground truth
  evaluate_handwriting.py        CER, WER and confidence for any sample
  live_check.py                  end-to-end run against a real backend and Ollama
tests/                           five modules, one per pipeline concern
sample_docs/                     digital and handwritten fixtures
  real_handwritten/              empty until a genuine human page is added
```

---

## Known limits

- TrOCR is English-only and reads one line at a time. Writing that runs across
  columns, or on a slanted baseline, degrades it.
- Recognition runs on CPU in this configuration, so a many-page scan is slow. A
  GPU would change the timings, not the design.
- Every accuracy figure above is **synthetic**. No real human handwriting has
  been measured, so no real-world accuracy is claimed.
- The answer verifier is the same local model as the generator. It is a
  fail-closed check, not a proof of correctness.
- DOCX tables are flattened during extraction, so cell alignment can be lost
  even when every value survives.
