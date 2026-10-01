#!/usr/bin/env python3
"""Enlarge the saved final MCMC legend without changing any fit drawing."""
from pathlib import Path
import shutil
import pymupdf as fitz

base = Path(__file__).resolve().parents[2]
path = base / 'S2223_Sim/output/MCMC_full_excitation_range_inputHist/all_MCMC_full_range_spectrum_individual_states_poisson_errors.pdf'
backup = base / 'tmp/pdfs/mcmc_legend_original' / path.name
backup.parent.mkdir(parents=True, exist_ok=True)
if not backup.exists():
    shutil.copy2(path, backup)
source = fitz.open(backup)
original = source[0]
# The original legend sits fully outside the plotting panels. Preserve the
# spectrum, residuals and inset as a vector drawing at their original size.
legend_left = 899.0
split_y = 266.5
legend_bottom = 529.0
scale = 1.4
legend_width = original.rect.width - legend_left
column_width = legend_width * scale
column_gap = 12.0
out = fitz.open()
page = out.new_page(width=legend_left + 2 * column_width + column_gap + 6,
                    height=original.rect.height)
plot_rect = fitz.Rect(0, 0, legend_left, original.rect.height)
page.show_pdf_page(plot_rect, source, 0, clip=plot_rect)
for column, (low, high) in enumerate(((0, split_y), (split_y, legend_bottom))):
    left = legend_left + column * (column_width + column_gap)
    top = 8.0 if column else 0.0
    destination = fitz.Rect(left, top, left + column_width, top + (high-low) * scale)
    clip = fitz.Rect(legend_left, low, original.rect.width, high)
    page.show_pdf_page(destination, source, 0, clip=clip)
# Ensure the fit panels render identically to the source.
a = original.get_pixmap(clip=plot_rect, alpha=False)
b = page.get_pixmap(clip=plot_rect, alpha=False)
assert a.samples == b.samples, 'Fit panel rendering changed'
temp = path.with_suffix('.updated.pdf')
out.save(temp, garbage=4, deflate=True)
out.close()
temp.replace(path)
check = fitz.open(path)
check[0].get_pixmap(matrix=fitz.Matrix(3,3), alpha=False).save(path.with_suffix('.png'))
print('Enlarged legend by 40%; fit, residuals, and inset verified pixel-identical.')
print(path)
