"""
Regenerate the handwritten samples in sample_docs/.

    python tools/make_handwritten_sample.py

The sample is rendered with 'Ink Free', a cursive font that ships with Windows,
so the repository can be tested end-to-end without using a real person's
handwriting and without downloading a dataset. Ground truth is written next to
the image, which makes character and word error rates measurable.

This is synthetic handwriting. Reported accuracy on it is not a claim about
accuracy on real human handwriting.
"""

import random
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_docs"

LINES = [
    "Section 1. Field notes on the water tank survey",
    "The survey covered twelve villages in the Kolar district.",
    "Each household received a 200 litre storage tank in June 2024.",
    "Water quality was tested at 46 sample points across the region.",
    "The report recommends a second survey before March 2025.",
]

SKEW_DEGREES = 2.5


def _font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("Inkfree.ttf", "segoesc.ttf"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise SystemExit("No handwritten font found. Install Windows 'Ink Free'.")


def _fit_font_size(draw: ImageDraw.ImageDraw, width: int, margin: int) -> int:
    """Shrink the script until the longest line fits between the margins.

    A fixed size was what clipped 'June 2024' and 'the region' off the right
    edge, so the size is derived from the measured text instead of assumed.
    """
    size = 40
    while size > 12:
        font = _font(size)
        widest = max(draw.textlength(line, font=font) for line in LINES)
        if widest <= width - 2 * margin:
            return size
        size -= 2
    raise SystemExit("Sample lines are too long for the rendered page.")


def render_png(path: Path) -> None:
    width, height, margin = 1400, 1900, 90
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    font = _font(_fit_font_size(draw, width, margin))
    rng = random.Random(7)

    y = 200
    for line in LINES:
        # Jitter each baseline so the page is not machine-straight.
        draw.text((margin, y + rng.randint(-4, 4)), line, font=font, fill=20)
        y += 120

    # expand=True keeps the whole rotated page; without it the corners of a
    # full-width line are cut off even when the unrotated text fit.
    image = image.rotate(SKEW_DEGREES, resample=Image.BICUBIC, expand=True, fillcolor=255)
    image.save(path)


def render_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=str(SAMPLES / "handwritten_notes.png"))

    typed = document.new_page(width=595, height=842)
    typed.insert_text((72, 90), "Appendix A. Typed summary of the survey", fontsize=12)
    typed.insert_text((72, 115), "Total tanks installed: 1,480.", fontsize=12)
    typed.insert_text((72, 135), "Budget used: 9.6 million rupees.", fontsize=12)

    document.save(str(path))
    document.close()


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    render_png(SAMPLES / "handwritten_notes.png")
    render_pdf(SAMPLES / "handwritten_notes.pdf")
    (SAMPLES / "handwritten_notes_ground_truth.txt").write_text(
        "\n".join(LINES) + "\n", encoding="utf-8"
    )
    print(f"Wrote 3 files to {SAMPLES}")


if __name__ == "__main__":
    main()
