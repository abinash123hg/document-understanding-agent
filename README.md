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
| 2 | Preprocessing: binarize, deskew, line segmentation (denoise is opt-in, off by default) | `backend/imaging.py` |
| 3 | Handwriting recognition (TrOCR) | `backend/handwriting.py` |
| 4 | Text normalization | `backend/processor.py` |
| 5 | Chunking | `backend/processor.py` |
| 6 | Embedding and indexing | `backend/retriever.py` |
| 7 | Retrieval: dense + BM25 + rerank | `backend/retriever.py` |
| 8 | Answer generation and verification | `backend/llm.py` |

Routing is decided per page, not per file:

1. A page with a real text layer is read with PyMuPDF and never goes near an OCR
   engine. Repeated header/footer lines and URL-only lines are dropped before
   anything is indexed, so the boilerplate cannot be retrieved and cited as
   content. A page that is *only* boilerplate and carries no image is dropped
   rather than sent to OCR - recognising it would hand the same header back
   unfiltered. A page that carries an image is still a scan, even when someone
   has typed a header over it, and still gets read.
2. A page with fewer than 20 embedded characters is rasterized and handed to
   **Tesseract first**. Printed scans are the case TrOCR is slowest and worst
   at, and Tesseract reads them in milliseconds.
3. Only a page Tesseract returns empty or garbled for goes to **TrOCR**, which
   is the handwriting model. Those chunks are tagged
   `content_type: "handwritten_ocr"`.

---

## Why TrOCR, and what Tesseract is for

- **TrOCR `microsoft/trocr-base-handwritten` is the handwriting model**, and the
  only one used for it. On the repository's synthetic sample its CER and WER
  matched the small model, with no measurable disadvantage, so the base model is
  the default and `trocr-small` is not offered as a shortcut. This does not
  establish how either model performs on real handwriting.
- **Tesseract is used, but only as a fast reader for printed scans.** Every
  rasterized page is offered to it first, and a page whose output comes back
  empty or mostly non-letter falls through to TrOCR - which is exactly what
  happens to handwriting, so handwritten pages are still recognised by TrOCR and
  still scored by it. The cost of that ordering is one extra Tesseract call per
  scanned page; the saving on a printed scan is far larger (measured below).
  `pytesseract` is in `requirements.txt`; the `tesseract-ocr` **binary is a
  separate install** (Windows: `winget install UB-Mannheim.TesseractOCR`).
  Without the binary, uploads still succeed - each printed page logs a warning
  and takes the slower TrOCR path.
- **Docling** was removed. It is a general document converter, and its layout
  machinery is dead weight once per-page handwriting recognition is the actual
  requirement. It was also the heaviest dependency in the project.

Line boundaries are found on the binarized mask, but the crops themselves are
taken from the grayscale image: thresholding throws away the stroke weight and
grey levels the recogniser relies on. Each line is then trimmed horizontally to
its own ink, avoiding unnecessary blank space around the text.

`fastNlMeansDenoising` is switched off by default (`DENOISE_IMAGES=false`). It
costs a full extra pass over every page image, and on these scans the deskew and
ink-trimmed line crops do the work it was meant to help with. Switching it on
does not remove either step.

---

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) running locally, with the answer model pulled
- The `tesseract-ocr` binary for the fast printed-scan path (optional - see above)
- Internet access **once**, on the first run, for the Hugging Face models

Everything here runs on CPU. There is no CUDA path in the code, and no GPU is
assumed anywhere in the timings below.

```bash
ollama pull qwen2.5:3b
```

The application never lets Ollama fetch a model on demand. If the configured
model is not already pulled, requests fail with a message naming the model and
the `ollama pull` command to run. A different model that happens to share its
name prefix is not accepted as a substitute; `qwen2.5:3b` and `qwen2.5:3b:latest`
are the same model to this check.

On Windows, install the Tesseract binary separately - the pip package is only a
wrapper that calls it:

```powershell
winget install UB-Mannheim.TesseractOCR
```

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
is 100 MB, enforced while streaming to disk, so an oversized file is rejected
before it has been fully read.

A `/chat` response contains `answer`, `document_name`, and `sources`. `sources`
is exactly the set of excerpts the answer was built from, in `[S1]..[Sn]` order:
`[S2]` in the answer text resolves to `sources[1]`, and no excerpt that the
answer stage rejected appears in it. That is why the UI can turn every label
into a link. A refusal returns `sources: []` and carries `refusal_reason`:
`no_evidence`, `low_handwriting_confidence`, `untraceable_citation`,
`unsupported_claim`, or `not_answerable`, so a refusal says which guard stopped
the answer rather than leaving the reading of it to guesswork.

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
   case below, the system never writes a label on the model's behalf. The one
   exception: when exactly one excerpt was supplied and the model numbered it
   wrong (`[S2]` against a single excerpt), the label is renumbered rather than
   refused - it cannot be naming evidence that was never shown, so there is
   nothing new to misattribute, and the verifier still has to approve the claim.
   A run of labels the model wrote side by side (`. [S1] [S2]`) collapses to the
   one chip that excerpt has. Refusing a correct answer over a label typo was
   measured as a real false negative on this machine, and the user reads it as
   the system never answering.
5. **Independent verifier.** A second pass checks that each claim is stated in
   the excerpts *and* that the draft answers the question asked. The drafting
   prompt asks for exactly that too - only what the question asks, no supported
   but unrelated background - because a real run padded a correct "200 litres"
   answer with the survey's village count and the verifier refused the whole
   thing. It fails closed: any error in the verifier means no answer.
6. **OCR confidence gate.** Each handwritten chunk carries the *minimum*
   confidence of the lines inside it, not the page average, so one unreadable
   line lowers only the chunk it landed in and cannot drag a clean chunk down
   with it. An excerpt is only shown to the model, and can only be cited, when
   it is digital text or handwriting at or above `HTR_MIN_CONFIDENCE`. Garbled
   handwriting is removed from the evidence rather than merely flagged, so a
   low-confidence line cannot be quoted as an answer; if nothing trustworthy
   remains, the system refuses.
7. **Temperature 0.0**, plus a prompt-injection rule: excerpts are data, never
   commands.

In the UI, every `[Sn]` label in a successful answer is rendered as a chip that
opens and highlights the matching source card. Labels are only turned into chips
when a source with that number exists, and a refusal renders none - so there is
no path where the page shows a citation that points at nothing.

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

Six modules: `test_handwriting.py`, `test_processor.py`, `test_retrieval.py`,
`test_llm.py`, `test_api.py`, `test_storage.py`. The one skip is
`test_recognition_on_a_real_handwritten_sample`, which stays skipped until a real
page is added (see below).

Measured in this run:

```
143 passed, 1 skipped                      (pytest -q)
1 passed, 1 skipped, 142 deselected        (pytest -m slow -q)
1 passed, 12 deselected                    (pytest tests/test_retrieval.py -k leak -q)
```

`tools/live_check.py` runs the same guarantees against a **real** running
backend, Ollama and TrOCR rather than mocks - upload, recognition, every `[Sn]`
label in an answer resolving to a returned source, refusal carrying no sources,
isolation, delete, re-upload, corrupt file, invalid file:

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

Same machine, CPU only, warm caches unless stated otherwise:

| Operation | Time |
|-----------|------|
| Extract generated 20-page digital PDF | 0.017 s |
| Warm full upload of that PDF (chunk + embed + index) | 2.484 s |
| Warm upload of the shipped handwritten sample PDF (1 scanned page, 1 text page) | 0.666 s |
| First TrOCR upload in a fresh process (includes loading the 1.3 GB weights) | 23.2 s |
| Upload of a generated 2-page **printed** scan, Tesseract fast path | 1.703 s |
| The same file with the fast path switched off, so TrOCR reads it | 391.5 s |

The last two rows are why the routing order is what it is: a printed page costs
TrOCR about three minutes and Tesseract under a second. A 40-page printed scan
left on the TrOCR path would run for hours, which is also the reason
`PDF_MAX_OCR_PAGES` (default 40) exists - a scan that large fails at once with a
message naming the page count and telling you to split the file, instead of
appearing to hang. `OCR_PAGES_AT_A_TIME` (default 2) caps how many pages are
rasterized and recognised at once, which is what keeps a big scan inside laptop
RAM.

A handwritten page pays for one wasted Tesseract attempt before TrOCR reads it,
and the tables above show that is cheap: the handwritten sample PDF uploads in
0.666 s with both engines on that page.

The digital and printed timing fixtures are generated by this run and contain no
real person's documents. All handwriting accuracy values are synthetic; accuracy
on real handwriting remains unmeasured.

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
tests/                           six modules, one per pipeline concern
sample_docs/                     digital and handwritten fixtures
  real_handwritten/              empty until a genuine human page is added
```

---

## Known limits

- TrOCR is English-only and reads one line at a time. Writing that runs across
  columns, or on a slanted baseline, degrades it.
- Recognition runs on CPU in this configuration, so a many-page scan is slow. A
  GPU would change the timings, not the design.
- The printed-scan fast path depends on the `tesseract-ocr` binary being
  installed; it is an external program, not a pip package. Without it every
  printed page takes the TrOCR path, which is correct but several times slower,
  and the backend logs a warning per page rather than failing the upload.
- Every accuracy figure above is **synthetic**. No real human handwriting has
  been measured, so no real-world accuracy is claimed.
- The answer verifier is the same local model as the generator. It is a
  fail-closed check, not a proof of correctness.
- DOCX tables are flattened during extraction, so cell alignment can be lost
  even when every value survives.
