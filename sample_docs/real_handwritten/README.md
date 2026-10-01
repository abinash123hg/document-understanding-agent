# Real handwritten samples

This folder is intentionally empty of images.

The benchmark the project reports is **synthetic**: `sample_docs/handwritten_notes.png`
is rendered with a cursive font, so it can never prove how the system behaves on a
real person's writing. Rather than guess at a real-world number, the test
`test_recognition_on_a_real_handwritten_sample` in `tests/test_handwriting.py`
stays skipped until someone adds:

- one page of genuine handwriting as `page1.png` (or `.jpg`)
- `ground_truth.txt` next to it — one transcription line per written line

Then measure it:

```bash
python tools/evaluate_handwriting.py \
  --image sample_docs/real_handwritten/page1.png \
  --ground-truth sample_docs/real_handwritten/ground_truth.txt \
  --label "Real handwritten sample"

pytest -m slow
```

Whatever CER, WER and confidence that page produces are reported as measured. A
messy page scores badly, and that result belongs in the README just as much as a
good one.

Upload a real page through the web UI to check the rest of the pipeline
(retrieval, grounded answers, citations) on it as well.
