"""
Pipeline stage 3: handwritten text recognition.

Uses TrOCR (microsoft/trocr-base-handwritten), a transformer trained
specifically on handwritten English, rather than a general printed-text OCR
engine. Returns the recognised text together with a mean token confidence so
downstream stages can flag low-quality pages instead of trusting them blindly.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from backend.config import settings

logger = logging.getLogger(__name__)


class HandwritingModelError(RuntimeError):
    """Raised when the TrOCR weights cannot be loaded."""


@lru_cache(maxsize=1)
def _model_and_processor():
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as error:
        raise HandwritingModelError(
            "transformers is not installed. Run: pip install -r requirements.txt"
        ) from error

    name = settings.htr_model
    try:
        processor = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name)
    except Exception as error:
        raise HandwritingModelError(
            f"Could not load the handwriting model '{name}'. "
            f"The first run downloads it from Hugging Face, so an internet "
            f"connection is required once. Original error: {error}"
        ) from error

    model.eval()
    logger.info("Loaded handwriting model %s", name)
    return model, processor


def is_loaded() -> bool:
    """Cheap: reports whether the weights are already in memory. Never loads
    them, so polling this endpoint cannot trigger a model download."""
    return _model_and_processor.cache_info().currsize > 0


def is_available() -> bool:
    try:
        _model_and_processor()
        return True
    except HandwritingModelError:
        return False


def _to_pil(crop) -> Image.Image:
    if isinstance(crop, Image.Image):
        return crop.convert("RGB")
    return Image.fromarray(np.asarray(crop)).convert("RGB")


def _sequence_confidence(scores, sequences, tokenizer) -> list[float]:
    """
    Mean probability the model assigned to each token it actually emitted.

    The returned sequence carries the forced decoder-start token ahead of the
    first scored step, so the token for scores[step] sits at column
    step + offset. Reading column `step` instead pairs every probability with
    the wrong token and collapses the result toward zero.
    """
    import torch

    offset = sequences.shape[1] - len(scores)
    skip = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    confidences = []

    for index in range(sequences.shape[0]):
        token_probs = []
        for step, step_scores in enumerate(scores):
            token_id = int(sequences[index, step + offset])
            if token_id in skip:
                continue
            probs = torch.softmax(step_scores[index], dim=-1)
            token_probs.append(float(probs[token_id]))

        confidences.append(
            sum(token_probs) / len(token_probs) if token_probs else 0.0
        )

    return confidences


def clean_line(text: str) -> str:
    """
    Drop the stray leading '#' TrOCR tends to emit as its very first token.

    On the shipped sample that token is the greedy argmax even though the model
    only gives it ~0.2 probability. Only a leading run of '#' is removed, so
    numbers, units, names and formulas elsewhere on the line survive untouched.
    """
    text = str(text or "").strip()
    while text.startswith("#"):
        text = text[1:].strip()
    return text


def recognize_lines(crops: list[np.ndarray]) -> list[tuple[str, float]]:
    """Recognise a list of single-line crops, returning (text, confidence) each."""
    if not crops:
        return []

    import torch

    model, processor = _model_and_processor()
    tokenizer = processor.tokenizer
    results: list[tuple[str, float]] = []
    batch_size = max(1, int(settings.trocr_line_batch))

    for start in range(0, len(crops), batch_size):
        batch = [_to_pil(crop) for crop in crops[start:start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt", padding=True)

        with torch.no_grad():
            output = model.generate(
                inputs.pixel_values,
                max_new_tokens=settings.htr_max_tokens,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        texts = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        scores = _sequence_confidence(output.scores, output.sequences, tokenizer)

        results.extend(
            (clean_line(text), confidence) for text, confidence in zip(texts, scores)
        )

    return results


def recognize_page_lines(image: np.ndarray) -> list[tuple[str, float]]:
    """
    Preprocess one page and recognise it line by line, keeping each confidence.

    The per-line scores are what let the chunker give a chunk the worst line it
    contains rather than an average a few clean lines can carry.
    """
    from backend import imaging

    crops, angle = imaging.preprocess_image(image)
    lines = recognize_lines(crops)

    logger.info(
        "Recognised %d lines, deskew %.2f deg, mean confidence %.3f",
        len(crops), angle, mean_confidence(lines),
    )
    return lines


def recognize_array(image: np.ndarray) -> tuple[str, float]:
    """Preprocess an in-memory image and recognise it. Used for PDF pages."""
    lines = recognize_page_lines(image)
    return _join_lines(lines), mean_confidence(lines)


def recognize_image(path: Path) -> tuple[str, float]:
    """Full preprocessing + recognition for one image file."""
    from backend import imaging

    return recognize_array(imaging.load_image(path))


def _join_lines(lines: list[tuple[str, float]]) -> str:
    return "\n".join(line for line, _ in lines if line.strip())


def mean_confidence(lines: list[tuple[str, float]]) -> float:
    kept = [confidence for line, confidence in lines if line.strip()]
    return sum(kept) / len(kept) if kept else 0.0

