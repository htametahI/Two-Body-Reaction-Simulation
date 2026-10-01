#!/usr/bin/env python3
"""Change the displayed 1808 gamma label to 1809 without changing histogram data."""
from pathlib import Path
import shutil
import pymupdf as fitz

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'S2223_Sim/output/inputHist_histogram_plots'
BACKUP = BASE / 'tmp/pdfs/gamma_gate_original_labels'
FONTS = BASE / '.venv/lib/python3.14/site-packages/matplotlib/mpl-data/fonts/ttf'
changed = []
for path in sorted(OUT.rglob('*.pdf')):
    doc = fitz.open(path)
    edits = []
    for page in doc:
        for block in page.get_text('rawdict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    chars = span['chars']
                    text = ''.join(c['c'] for c in chars)
                    if '1808' not in text:
                        continue
                    index = text.index('1808') + 3
                    char = chars[index]
                    edits.append((page.number, char, span['font'], span['size'], span['color']))
    if not edits:
        doc.close()
        continue
    backup = BACKUP / path.relative_to(OUT)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    for number, char, font, size, color in edits:
        box = fitz.Rect(char['bbox'])
        # Interior rectangle selects only the final digit, preserving adjacent glyphs.
        box.x0 += .15; box.x1 -= .15
        doc[number].add_redact_annot(box, fill=False)
    for number in sorted({e[0] for e in edits}):
        doc[number].apply_redactions(images=0, graphics=0)
    for number, char, font, size, color in edits:
        fontfile = FONTS / ('cmr10.ttf' if font == 'Cmr10' else 'DejaVuSerif.ttf')
        assert font in ('Cmr10', 'DejaVuSerif'), font
        page = doc[number]
        page.insert_font(fontname='CorrectedGammaLabel' + font, fontfile=str(fontfile))
        page.insert_text(char['origin'], '9', fontsize=size,
                         fontname='CorrectedGammaLabel' + font,
                         color=fitz.sRGB_to_pdf(color))
    temporary = path.with_suffix('.updated.pdf')
    doc.save(temporary, garbage=4, deflate=True)
    doc.close()
    temporary.replace(path)
    check = fitz.open(path)
    assert all('1808' not in p.get_text() for p in check)
    pages = sorted({e[0] for e in edits})
    if len(check) == 1:
        png = OUT / 'individual' / (path.stem + '.png') if path.parent.name == 'individual_pdf' else path.with_suffix('.png')
        check[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(png)
    for number in pages:
        preview = BASE / 'tmp/pdfs/gamma_gate_label_review' / f'{path.stem}_{number}.png'
        preview.parent.mkdir(parents=True, exist_ok=True)
        check[number].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).save(preview)
    check.close()
    changed.append(str(path.relative_to(OUT)))
print('\n'.join(changed))
