"""Shared ReportLab styles and helpers for the report and the beginner's guide.

Build with:  uv run --with reportlab python report/build_report.py
             uv run --with reportlab python report/build_guide.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
LOGS = ROOT / "logs"

STUDENT = "Mahamat Bakhit Tidjani"
GITHUB = "https://github.com/Mahamatbt/Mahamat_rl_summative"
VIDEO = "[add your video link here]"

# Fonts with Greek glyphs (gamma, lambda, epsilon) - fall back to Helvetica elsewhere
FONT, FONT_B, FONT_I = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
for name, files in {
    "Body": ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"],
    "BodyB": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"],
    "BodyI": ["/System/Library/Fonts/Supplemental/Arial Italic.ttf", "/Library/Fonts/Arial Italic.ttf", "C:/Windows/Fonts/ariali.ttf"],
}.items():
    for f in files:
        if Path(f).exists():
            pdfmetrics.registerFont(TTFont(name, f))
            if name == "Body":
                FONT = "Body"
            elif name == "BodyB":
                FONT_B = "BodyB"
            else:
                FONT_I = "BodyI"
            break
if FONT == "Body" and FONT_B == "BodyB":
    from reportlab.pdfbase.pdfmetrics import registerFontFamily

    registerFontFamily("Body", normal="Body", bold="BodyB", italic=FONT_I, boldItalic="BodyB")

INK = colors.HexColor("#1d2430")
MUTED = colors.HexColor("#5b6472")
ACCENT = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#d9dde3")
SOFT = colors.HexColor("#f2f5f9")
NOTE_BG = colors.HexColor("#eef5ff")
WARN_BG = colors.HexColor("#fff6e5")


def styles(base_size: float = 10):
    ss = getSampleStyleSheet()
    lead = base_size * 1.38
    s = {
        "title": ParagraphStyle("title", fontName=FONT_B, fontSize=20, leading=25, textColor=INK, spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", fontName=FONT, fontSize=11, leading=15, textColor=MUTED, spaceAfter=4),
        "h1": ParagraphStyle("h1", fontName=FONT_B, fontSize=base_size + 4.5, leading=base_size + 8, textColor=ACCENT, spaceBefore=10, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName=FONT_B, fontSize=base_size + 1.5, leading=base_size + 5, textColor=INK, spaceBefore=7, spaceAfter=3),
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontName=FONT, fontSize=base_size, leading=lead, textColor=INK, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", fontName=FONT, fontSize=base_size, leading=lead, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=1.5),
        "small": ParagraphStyle("small", fontName=FONT, fontSize=base_size - 2, leading=(base_size - 2) * 1.3, textColor=MUTED),
        "caption": ParagraphStyle("caption", fontName=FONT_I, fontSize=base_size - 1.5, leading=(base_size - 1.5) * 1.3, textColor=MUTED, alignment=TA_CENTER, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName=FONT, fontSize=base_size - 2.2, leading=(base_size - 2.2) * 1.25, textColor=INK),
        "cellb": ParagraphStyle("cellb", fontName=FONT_B, fontSize=base_size - 2.2, leading=(base_size - 2.2) * 1.25, textColor=colors.white),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=base_size - 1.5, leading=(base_size - 1.5) * 1.35, textColor=INK, backColor=SOFT, borderPadding=5, leftIndent=6, spaceBefore=3, spaceAfter=7),
    }
    return s


def P(text, style):
    return Paragraph(text, style)


def bullets(items, st):
    return [Paragraph(t, st["bullet"], bulletText="•") for t in items]


def figure(path: Path, width_cm: float, caption: str | None, st):
    from reportlab.lib.utils import ImageReader

    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    img = Image(str(path), width=w, height=w * ih / iw)
    parts = [img]
    if caption:
        parts.append(Paragraph(caption, st["caption"]))
    return KeepTogether(parts)


def table(rows, col_widths, st, header=True, highlight_row: int | None = None, zebra=True, align_right_from: int | None = None):
    data = []
    for i, r in enumerate(rows):
        style = st["cellb"] if (header and i == 0) else st["cell"]
        data.append([c if not isinstance(c, str) else Paragraph(c, style) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), INK)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), SOFT))
    if highlight_row is not None:
        cmds.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#dcebff")))
    t.setStyle(TableStyle(cmds))
    return t


def callout(text, st, bg=NOTE_BG, border=ACCENT):
    t = Table([[Paragraph(text, st["body"])]], colWidths=["100%"])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("LINEBEFORE", (0, 0), (0, -1), 3, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return KeepTogether([t, Spacer(1, 6)])


def read_summary(algo: str) -> list[dict]:
    with open(LOGS / algo / "summary.csv", newline="") as f:
        return sorted(csv.DictReader(f), key=lambda r: r["run"])


def read_comparison() -> dict:
    with open(LOGS / "comparison.csv", newline="") as f:
        return {r["agent"]: r for r in csv.DictReader(f)}


def read_metrics() -> dict:
    return json.loads((LOGS / "report_metrics.json").read_text())


def best_run(algo: str) -> str:
    sub = "dqn" if algo == "dqn" else "pg"
    return json.loads((ROOT / "models" / sub / f"best_{algo}.json").read_text())["run"]


def page_decor(title: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MUTED)
        w, h = doc.pagesize
        canvas.drawString(doc.leftMargin, 1.1 * cm, title)
        canvas.drawRightString(w - doc.rightMargin, 1.1 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(doc.leftMargin, 1.45 * cm, w - doc.rightMargin, 1.45 * cm)
        canvas.restoreState()

    return draw
