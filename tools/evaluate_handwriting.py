"""
Score handwritten-text recognition on one sample.

    # the synthetic page shipped with the repository
    python tools/evaluate_handwriting.py

    # a real handwritten page, once one is available
    python tools/evaluate_handwriting.py \
        --image sample_docs/real_handwritten/page1.png \
        --ground-truth sample_docs/real_handwritten/ground_truth.txt \
        --label "Real handwritten sample"

Prints the recognised lines exactly as the pipeline returns them, then CER, WER
and the recogniser's own mean token confidence. Nothing here adjusts a number:
whatever TrOCR emits is what gets scored, so a poor real sample would report a
poor score.

Ground truth is one line per written line. CER is character-level Levenshtein
over the whole page; WER is word-level Levenshtein. Both are recogniser metrics
only and say nothing about retrieval or answer quality.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_IMAGE = ROOT / "sample_docs/handwritten_notes.png"
DEFAULT_TRUTH = ROOT / "sample_docs/handwritten_notes_ground_truth.txt"


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _normalize(lines):
    """Lowercase and collapse all whitespace to single spaces.

    Spaces are kept in the character stream on purpose: dropping them would make
    CER insensitive to word-boundary errors and report a flattering zero.
    """
    return " ".join(" ".join(lines).lower().split())


def score(truth_lines, hypothesis_lines):
    truth, hypothesis = _normalize(truth_lines), _normalize(hypothesis_lines)
    cer = edit_distance(truth, hypothesis) / max(1, len(truth))
    truth_words, hypothesis_words = truth.split(), hypothesis.split()
    wer = edit_distance(truth_words, hypothesis_words) / max(1, len(truth_words))
    return cer, wer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--label", default="Synthetic handwriting benchmark")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"No such image: {args.image}")
    if not args.ground_truth.exists():
        raise SystemExit(f"No such ground truth: {args.ground_truth}")

    from backend import handwriting

    if not handwriting.is_available():
        raise SystemExit(
            "TrOCR weights are not cached locally. Run once with internet access "
            "so microsoft/trocr-base-handwritten is downloaded."
        )

    truth_lines = args.ground_truth.read_text(encoding="utf-8").strip().splitlines()
    text, confidence = handwriting.recognize_image(args.image)
    hypothesis_lines = text.strip().splitlines()

    cer, wer = score(truth_lines, hypothesis_lines)

    print(f"Sample      : {args.label}")
    print(f"Image       : {args.image}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Lines       : {len(truth_lines)} expected, {len(hypothesis_lines)} recognised")
    print()
    print("--- recognised ---")
    for line in hypothesis_lines:
        print(line)
    print()
    print(f"CER        : {cer:.3f}")
    print(f"WER        : {wer:.3f}")
    print(f"Confidence : {confidence:.3f}")


if __name__ == "__main__":
    main()
