"""
Stage 1-3 tests: image preprocessing and handwriting recognition.

CER and WER measured here describe the recogniser only. They are deliberately
kept apart from retrieval and answer accuracy, which are separate measurements
of separate stages.
"""

import math

import numpy as np
import pytest

from backend import handwriting, imaging


def drawn_page(lines=5, skew=0.0, size=(700, 900)):
    import cv2

    height, width = size
    image = np.full((height, width), 255, dtype=np.uint8)

    for row in range(lines):
        y = 90 + row * 95
        for word in range(4):
            x = 80 + word * 190
            cv2.putText(image, "handwritten", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, 25, -1, cv2.LINE_AA)

    if skew:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), skew, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), borderValue=255)

    return image


@pytest.mark.parametrize("skew", [-8.0, -3.0, 3.0, 8.0])
def test_deskew_removes_the_applied_skew(skew):
    straight, reported = imaging.deskew(imaging.to_gray(drawn_page(skew=skew)))
    residual = imaging.skew_angle(imaging.binarize(straight))

    assert reported == pytest.approx(-skew, abs=0.6), "deskew must rotate the other way"
    assert abs(residual) < 0.5


def test_deskew_leaves_a_straight_page_alone():
    gray = imaging.to_gray(drawn_page())

    out, angle = imaging.deskew(gray)

    assert angle == 0.0
    assert np.array_equal(out, gray)


def test_extreme_skew_is_refused_rather_than_mangled():
    """A page tilted past the trustworthy limit is better left untouched than
    rotated into something the recogniser cannot read."""
    gray = imaging.to_gray(drawn_page(skew=40.0))

    assert imaging.skew_angle(imaging.binarize(gray)) == 0.0


def test_line_segmentation_finds_every_drawn_line():
    crops = imaging.segment_lines(imaging.to_gray(drawn_page(lines=5)))

    assert len(crops) == 5
    assert all(crop.shape[0] >= imaging.MIN_LINE_HEIGHT for crop in crops)


def test_crops_keep_grey_levels_from_the_original_image():
    """Thresholding is used to locate lines, never to feed them to TrOCR."""
    gray = imaging.to_gray(drawn_page())
    crop = imaging.segment_lines(gray)[0]

    intermediate = np.count_nonzero((crop > 30) & (crop < 225))

    assert intermediate > 0


def test_blank_image_falls_back_to_a_single_crop():
    crops = imaging.segment_lines(np.full((200, 200), 255, dtype=np.uint8))

    assert len(crops) == 1


def test_runs_reports_contiguous_true_ranges():
    flags = np.array([False, True, True, False, False, True])

    assert imaging._runs(flags) == [(1, 3), (5, 6)]
    assert imaging._runs(np.zeros(4, dtype=bool)) == []


def test_hairline_rows_are_not_treated_as_text():
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[100:102, :] = 255

    assert imaging.line_bands(mask) == []


def test_clean_line_only_strips_a_leading_hash():
    assert handwriting.clean_line("# Section 1. Tank 200L") == "Section 1. Tank 200L"
    assert handwriting.clean_line("##  pH=7.2 #4") == "pH=7.2 #4"
    assert handwriting.clean_line("  46 sample points ") == "46 sample points"


def test_recognize_lines_handles_no_crops():
    assert handwriting.recognize_lines([]) == []


def test_confidence_pairs_each_score_with_the_token_it_generated():
    """The sequence carries the forced decoder-start token ahead of the first
    scored step. Reading the same column index as the scores collapses the
    result toward zero, which is the bug this test pins down."""
    import torch

    scores = [
        torch.tensor([[0.0, 9.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 9.0, 0.0, 0.0]]),
    ]
    sequences = torch.tensor([[0, 1, 2]])

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 3

    first = float(torch.softmax(scores[0][0], dim=-1)[1])
    second = float(torch.softmax(scores[1][0], dim=-1)[2])

    confidence = handwriting._sequence_confidence(scores, sequences, Tokenizer())

    assert confidence == pytest.approx([(first + second) / 2])
    assert confidence[0] > 0.9


def test_mean_confidence_ignores_empty_lines():
    assert handwriting.mean_confidence([("a", 0.8), ("", 0.1), ("  ", 0.0)]) == pytest.approx(0.8)
    assert handwriting.mean_confidence([]) == 0.0


def test_join_lines_keeps_line_order():
    assert handwriting._join_lines([("beta", 0.9), ("", 0.5), ("alpha", 0.7)]) == "beta\nalpha"


def test_model_error_explains_the_first_download(monkeypatch):
    def explode():
        raise RuntimeError("offline")

    import transformers

    monkeypatch.setattr(transformers.TrOCRProcessor, "from_pretrained",
                        classmethod(lambda cls, *a, **k: explode()))

    handwriting._model_and_processor.cache_clear()

    with pytest.raises(handwriting.HandwritingModelError, match="Hugging Face"):
        handwriting._model_and_processor()

    handwriting._model_and_processor.cache_clear()


def edit_distance(a, b):
    """Levenshtein over any sequence - characters for CER, words for WER."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalize(lines):
    """Lowercase and collapse whitespace, but keep the spaces themselves:
    dropping them would hide word-boundary errors and report a flattering zero."""
    return " ".join(" ".join(lines).lower().split())


def error_rates(truth_lines, hypothesis_lines):
    truth = normalize(truth_lines)
    hypothesis = normalize(hypothesis_lines)
    cer = edit_distance(truth, hypothesis) / max(1, len(truth))
    wer = edit_distance(truth.split(), hypothesis.split()) / max(
        1, len(truth.split())
    )
    return cer, wer


def sample_dir():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "sample_docs"


@pytest.mark.slow
def test_recognition_accuracy_on_the_shipped_sample():
    """Synthetic handwriting rendered with a cursive font. These numbers
    describe that page, not real human handwriting."""
    if not handwriting.is_available():
        pytest.skip("TrOCR weights are not cached; run once with internet access")

    root = sample_dir()
    truth_lines = (root / "handwritten_notes_ground_truth.txt").read_text(
        encoding="utf-8"
    ).strip().splitlines()

    text, confidence = handwriting.recognize_image(root / "handwritten_notes.png")
    hypothesis_lines = text.strip().splitlines()

    cer, wer = error_rates(truth_lines, hypothesis_lines)

    assert len(hypothesis_lines) == len(truth_lines)
    assert confidence > 0.5
    assert cer < 0.15
    assert wer < 0.25
    assert not any(line.startswith("#") for line in hypothesis_lines)
    assert math.isfinite(cer)
    assert math.isfinite(wer)


@pytest.mark.slow
def test_recognition_on_a_real_handwritten_sample():
    """
    The same measurement on genuine human handwriting.

    This repository ships no person's writing, so the test stays skipped until a
    page plus its transcription are dropped into sample_docs/real_handwritten/.
    No real-world accuracy figure is claimed anywhere until that happens, which
    is why there is no quality threshold here: the thresholds below only check
    that the measurement itself is valid, whatever score a real page earns.
    """
    root = sample_dir() / "real_handwritten"
    images = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg"))
    truth_file = root / "ground_truth.txt"

    if not images or not truth_file.exists():
        pytest.skip(
            "No real handwritten sample with a ground truth transcription; "
            "the project reports synthetic numbers only."
        )
    if not handwriting.is_available():
        pytest.skip("TrOCR weights are not cached; run once with internet access")

    truth_lines = truth_file.read_text(encoding="utf-8").strip().splitlines()
    text, confidence = handwriting.recognize_image(images[0])
    hypothesis_lines = text.strip().splitlines()

    cer, wer = error_rates(truth_lines, hypothesis_lines)

    assert hypothesis_lines
    assert 0.0 <= cer <= 1.0
    assert 0.0 <= wer <= 1.0
    assert 0.0 < confidence <= 1.0
