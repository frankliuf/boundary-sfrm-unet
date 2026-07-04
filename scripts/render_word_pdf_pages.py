from pathlib import Path

import pypdfium2 as pdfium


PDF = Path(r"D:\paper_MedIA Vol. 107–113\failure_region_reliability\manuscript\word_draft\qa_render\paper1_sfrm_audit_draft_word.pdf")
OUT = PDF.parent / "pages"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("page-*.png"):
        old.unlink()
    pdf = pdfium.PdfDocument(PDF)
    scale = 150 / 72
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image.save(OUT / f"page-{index + 1:02d}.png", "PNG")
        page.close()
    pdf.close()
    print(f"pages={len(list(OUT.glob('page-*.png')))} out={OUT}")


if __name__ == "__main__":
    main()
