"""Report assembly: sections in a fixed order, every generated figure, and an appendix of the rest.

Content is recorded as plain data and turned into reportlab flowables (PDF) or HTML
(fitfleet.visualization.html) only when a file is built, so both formats show exactly the
same report and a failed PDF build can be retried text-only. Files are written to a
temporary file and moved into place, so a failure never leaves a truncated report.
"""
from __future__ import annotations

import datetime
import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, NamedTuple, Sequence
from xml.sax.saxutils import escape

from fitfleet.utils.enums import ReportFormat
from fitfleet.visualization.core import DPI, SECTIONS, VisualizationReport, VizStatus

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("fitfleet.visualization")

FRONT = ("Executive Summary", "Experiment Configuration", "Dataset Overview")
APPENDIX = ("Visualization Index", "Errors, Warnings and Missing Visualizations")
EXECUTION = ("Execution Summary",)  # stage-by-stage PASS / FAIL of the run (fitfleet.tracking)
ORDER = FRONT + SECTIONS + EXECUTION + APPENDIX

ACCENT = "#1c5cab"
INK = "#0b0b0b"
MUTED = "#52514e"
RULE = "#d9d8d3"
ZEBRA = "#f5f5f3"
STATUS_INK = {VizStatus.SUCCESS: "#006300", VizStatus.SKIPPED: "#7a5a00", VizStatus.FAILED: "#b42318"}

# explanations shared by the PDF and the HTML appendix
FAILED_NOTE = ("These plots raised an error. The experiment and every other plot were not "
               "affected; the error is recorded here and in the fitfleet.visualization log.")
SKIPPED_NOTE = "Not applicable to the data or results available - nothing was estimated to fill the gap."


def _text(value: Any) -> str:
    """Escape for reportlab's paragraph markup."""
    return escape(str(value))


def format_cell(value: Any, digits: int = 4) -> str:
    """A table cell as text: floats with `digits` decimals, NaN as empty, the rest as str()."""
    import numpy as np
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def frame_rows(frame: pd.DataFrame, index: bool, digits: int) -> tuple[list[str], list[list[str]]]:
    """(header, rows) of a DataFrame as text, the index first when `index`."""
    columns = ([frame.index.name or ""] if index else []) + [str(c) for c in frame.columns]
    rows = []
    for idx, row in frame.astype(object).iterrows():  # object: integer columns stay integers
        rows.append(([format_cell(idx, digits)] if index else []) + [format_cell(v, digits) for v in row.tolist()])
    return columns, rows


def figure_meta(result, identifiers: dict[str, str]) -> str:
    """The line under a figure's title: its id and the model(s) it shows."""
    models = [f"{m} ({identifiers[m]})" if m in identifiers else m for m in result.models]
    return "  ·  ".join([f"id: {result.key}"] + ([f"model(s): {', '.join(models)}"] if models else []))


INDEX_HEADER = ("#", "visualization", "section", "status", "time (s)", "detail")


def index_intro(viz: VisualizationReport) -> str:
    """The sentence above the visualization index."""
    return f"Every visualization attempted for this report: {viz.summary()}."


def index_rows(viz: VisualizationReport) -> list[tuple[int, Any, str]]:
    """(number, result, detail) for each row of the visualization index."""
    return [(i, r, ("included in the report" if r.ok else r.reason)[:300]) for i, r in enumerate(viz, start=1)]


class Section(NamedTuple):
    """One numbered section of a built report, with what goes in it."""
    number: int
    name: str
    blocks: list            # (kind, payload) as recorded by paragraph() / table() / ...
    figures: list           # SUCCESS VisualizationResults placed in this section
    new_page: bool          # the PDF starts it on a new page (so does the HTML, when printed)


class PdfReport:
    """Collects text, tables and figures per section and builds the PDF.

    Sections without content are left out. Figures come from a VisualizationReport:
    every SUCCESS result is placed in its section; skipped and failed ones are listed
    in the appendix together with any warning raised while building the PDF.
    """

    def __init__(self, title: str, subtitle: str = "", landscape_pages: bool = True) -> None:
        self.title = title
        self.subtitle = subtitle
        self.landscape = landscape_pages
        self.created = datetime.datetime.now()
        self._blocks: dict[str, list[tuple[str, Any]]] = {name: [] for name in ORDER}
        self.warnings: list[str] = []
        self.images_embedded = 0

    # ------------------------------------------------------------ content
    def _add(self, section: str, kind: str, payload: Any) -> PdfReport:
        if section not in self._blocks:
            raise ValueError(f"unknown section {section!r}")
        self._blocks[section].append((kind, payload))
        return self

    def paragraph(self, section: str, text: str) -> PdfReport:
        return self._add(section, "paragraph", text)

    def heading(self, section: str, text: str) -> PdfReport:
        return self._add(section, "heading", text)

    def bullets(self, section: str, items: Iterable[str]) -> PdfReport:
        return self._add(section, "bullets", list(items))

    def key_values(self, section: str, rows: Sequence[tuple[str, Any]]) -> PdfReport:
        return self._add(section, "kv", list(rows))

    def table(self, section: str, frame: pd.DataFrame, index: bool = True, max_rows: int | None = None,
              digits: int = 4) -> PdfReport:
        frame = frame.head(max_rows) if max_rows else frame
        return self._add(section, "table", (frame, index, digits))

    def preformatted(self, section: str, text: str) -> PdfReport:
        return self._add(section, "pre", text)

    def sections(self, viz: VisualizationReport, images: bool = True) -> list[Section]:
        """The sections that have content, numbered, in ORDER. Both formats are built from this.

        `images` False: the figures are left out (a PDF rebuilt text-only), so the appendix
        that explains it is included."""
        figures: dict[str, list] = {name: [] for name in SECTIONS}
        for result in viz.succeeded:
            figures.setdefault(result.section, []).append(result)
        appendix_needed = len(viz) > 0 or self.warnings or not images
        out: list[Section] = []
        for name in ORDER:
            blocks = self._blocks.get(name, [])
            section_figures = figures.get(name, [])
            if name == APPENDIX[0]:
                has_content = len(viz) > 0
            elif name == APPENDIX[1]:
                has_content = appendix_needed and (viz.skipped or viz.failed or self.warnings or not images)
            else:
                has_content = bool(blocks or section_figures)
            if not has_content:
                continue
            new_page = bool(out) and bool(section_figures or name in APPENDIX or name in EXECUTION
                                          or name == SECTIONS[0])
            out.append(Section(len(out) + 1, name, blocks, section_figures, new_page))
        return out

    def side_by_side(self, result, images: bool = True) -> bool:
        """Landscape layout: the figure on the left, its explanation and values on the right.
        Wide figures (small multiples, metric panels) need the full width."""
        w_px, h_px = result.size_px or (1200, 800)
        return bool(self.landscape and images and result.image and len(result.metrics) <= 14
                    and w_px / h_px < 2.0)

    # ------------------------------------------------------------ build
    def build(self, path: str | Path, visualizations: VisualizationReport | None = None,
              identifiers: dict[str, str] | None = None) -> Path:
        """Write the PDF to `path`. If the figures can't be embedded, retry without them."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        visualizations = visualizations or VisualizationReport()
        partial = path.with_name(path.name + ".part")
        try:
            try:
                self._build(partial, visualizations, identifiers or {}, images=True)
            except Exception as e:
                message = f"embedding figures failed ({type(e).__name__}: {e}); report rebuilt without figures"
                logger.warning(message)
                self.warnings.append(message)
                self._build(partial, visualizations, identifiers or {}, images=False)
            os.replace(partial, path)
        finally:
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass
        return path

    def build_html(self, path: str | Path, visualizations: VisualizationReport | None = None,
                   identifiers: dict[str, str] | None = None) -> Path:
        """Write the same report as one self-contained HTML file (figures embedded) to `path`."""
        from fitfleet.visualization.html import render_report, write_atomic
        page = render_report(self, visualizations or VisualizationReport(), identifiers or {})
        return write_atomic(Path(path), page)

    def write(self, stem: str | Path, report_format: ReportFormat | str = ReportFormat.pdf,
              visualizations: VisualizationReport | None = None,
              identifiers: dict[str, str] | None = None) -> list[Path]:
        """Write the report to `stem`.pdf, `stem`.html or both (ReportFormat.pdf / .html / .dual).

        Returns the paths, PDF first. A warning raised while building one file (such as a PDF
        rebuilt without figures) is reported in that file only."""
        stem = Path(stem)
        warnings = list(self.warnings)
        paths = []
        for fmt in ReportFormat.parse(report_format).files:
            self.warnings = list(warnings)
            path = stem.with_name(f"{stem.name}.{fmt}")
            paths.append(self.build(path, visualizations, identifiers) if fmt == "pdf"
                         else self.build_html(path, visualizations, identifiers))
        return paths

    def _build(self, path: Path, viz: VisualizationReport, identifiers: dict[str, str], images: bool) -> None:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate

        pagesize = landscape(A4) if self.landscape else A4
        doc = SimpleDocTemplate(str(path), pagesize=pagesize, title=self.title, author="fitfleet",
                                leftMargin=36, rightMargin=36, topMargin=42, bottomMargin=36)
        self.images_embedded = 0
        story = self._story(doc, viz, identifiers, images)
        doc.build(story, onFirstPage=self._decorate, onLaterPages=self._decorate)

    def _decorate(self, canvas, doc) -> None:
        width, height = doc.pagesize
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, height - 26, self.title)
        canvas.drawRightString(width - doc.rightMargin, height - 26, f"{self.created:%Y-%m-%d %H:%M}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, height - 30, width - doc.rightMargin, height - 30)
        canvas.drawRightString(width - doc.rightMargin, 20, f"page {doc.page}")
        canvas.restoreState()

    # ------------------------------------------------------------ story
    def _styles(self):
        from reportlab.lib.colors import HexColor
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        base = getSampleStyleSheet()
        return {
            "title": ParagraphStyle("T", parent=base["Title"], fontSize=20, leading=24, alignment=0,
                                    textColor=HexColor(INK), spaceAfter=4),
            "subtitle": ParagraphStyle("ST", parent=base["Normal"], fontSize=10, textColor=HexColor(MUTED),
                                       spaceAfter=14),
            "h1": ParagraphStyle("H1", parent=base["Heading1"], fontSize=15, leading=18, textColor=HexColor(ACCENT),
                                 spaceBefore=4, spaceAfter=8),
            "h2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=11.5, leading=14, spaceBefore=8,
                                 spaceAfter=4, textColor=HexColor(INK)),
            "h3": ParagraphStyle("H3", parent=base["Heading3"], fontSize=10.5, leading=13, spaceBefore=4,
                                 spaceAfter=2, textColor=HexColor(INK)),
            "body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9, leading=12),
            "meta": ParagraphStyle("M", parent=base["BodyText"], fontSize=7.5, leading=9.5, textColor=HexColor(MUTED)),
            "caption": ParagraphStyle("C", parent=base["BodyText"], fontSize=8, leading=10.5, textColor=HexColor(MUTED),
                                      spaceBefore=3),
            "cell": ParagraphStyle("Cell", parent=base["BodyText"], fontSize=7, leading=8.5),
            "cellhead": ParagraphStyle("CellH", parent=base["BodyText"], fontSize=7, leading=8.5,
                                       fontName="Helvetica-Bold"),
            "pre": ParagraphStyle("Pre", parent=base["Code"], fontSize=7, leading=8.5),
        }

    def _story(self, doc, viz: VisualizationReport, identifiers: dict[str, str], images: bool) -> list:
        from reportlab.platypus import PageBreak, Paragraph, Spacer
        styles = self._styles()
        width, height = doc.width, doc.height
        story = [Paragraph(_text(self.title), styles["title"])]
        if self.subtitle:
            story.append(Paragraph(_text(self.subtitle), styles["subtitle"]))

        for section in self.sections(viz, images):
            if section.new_page:
                story.append(PageBreak())
            elif section.number > 1:
                story.append(Spacer(1, 10))
            story.append(Paragraph(f"{section.number}. {_text(section.name)}", styles["h1"]))
            for kind, payload in section.blocks:
                story.extend(self._block(kind, payload, styles, width))
            for result in section.figures:
                story.extend(self._figure(result, styles, width, height, identifiers, images))
            if section.name == APPENDIX[0]:
                story.extend(self._index(viz, styles, width))
            elif section.name == APPENDIX[1]:
                story.extend(self._problems(viz, styles, images))
        return story

    def _block(self, kind: str, payload: Any, styles, width: float) -> list:
        from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, Spacer
        if kind == "paragraph":
            return [Paragraph(_text(payload), styles["body"]), Spacer(1, 4)]
        if kind == "heading":
            return [Paragraph(_text(payload), styles["h2"])]
        if kind == "bullets":
            return [ListFlowable([ListItem(Paragraph(_text(i), styles["body"]), leftIndent=10) for i in payload],
                                 bulletType="bullet", start="•", leftIndent=10), Spacer(1, 4)]
        if kind == "kv":
            return [self._kv_table(payload, styles, min(width, 520)), Spacer(1, 6)]
        if kind == "table":
            frame, index, digits = payload
            return [self._frame_table(frame, index, digits, styles, width), Spacer(1, 6)]
        if kind == "pre":
            return [Preformatted(str(payload), styles["pre"]), Spacer(1, 4)]
        raise ValueError(kind)

    @staticmethod
    def _table_style(header: bool):
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import TableStyle
        commands = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.25, HexColor(RULE)),
                    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
        if header:
            commands += [("LINEBELOW", (0, 0), (-1, 0), 0.8, HexColor(MUTED)),
                         ("ROWBACKGROUNDS", (0, 1), (-1, -1), [None, HexColor(ZEBRA)])]
        return TableStyle(commands)

    def _kv_table(self, rows, styles, width: float):
        from reportlab.platypus import Paragraph, Table
        data = [[Paragraph(_text(k), styles["cellhead"]), Paragraph(_text(v), styles["cell"])] for k, v in rows]
        table = Table(data, colWidths=[width * 0.36, width * 0.64], hAlign="LEFT")
        table.setStyle(self._table_style(header=False))
        return table

    def _frame_table(self, frame, index: bool, digits: int, styles, width: float):
        from reportlab.platypus import Paragraph, Table

        columns, rows = frame_rows(frame, index, digits)
        # Headers are bold (wider glyphs): weight them up so "mean_f1" doesn't wrap as "mean_f / 1".
        lengths = [max([round(len(columns[i]) * 1.2)] + [len(r[i]) for r in rows]) for i in range(len(columns))]
        weights = [min(max(n, 4), 48) for n in lengths]
        widths = [width * w / sum(weights) for w in weights]
        data = [[Paragraph(_text(c), styles["cellhead"]) for c in columns]]
        data += [[Paragraph(_text(v), styles["cell"]) for v in r] for r in rows]
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(self._table_style(header=True))
        return table

    def _figure(self, result, styles, width: float, height: float, identifiers: dict[str, str], images: bool) -> list:
        from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle
        # spacing goes before the title, inside the block: a trailing spacer could spill onto
        # an otherwise empty page just before a section's page break
        parts = [Spacer(1, 10), Paragraph(_text(result.title), styles["h2"]),
                 Paragraph(_text(figure_meta(result, identifiers)), styles["meta"])]
        caption = [Paragraph("<b>What it shows.</b> " + _text(result.description), styles["caption"])]
        if result.note:
            caption.append(Paragraph(_text(result.note), styles["caption"]))
        if self.side_by_side(result, images):  # landscape: figure left, explanation and values right
            w_px, h_px = result.size_px or (1200, 800)
            w_pt, h_pt = w_px / DPI * 72, h_px / DPI * 72
            left, right = width * 0.64, width * 0.34
            scale = min(left / w_pt, (height * 0.78) / h_pt, 1.0)
            image = Image(io.BytesIO(result.image), width=w_pt * scale, height=h_pt * scale)
            self.images_embedded += 1
            side = caption + ([Spacer(1, 6), self._kv_table(list(result.metrics.items()), styles, right - 8)]
                              if result.metrics else [])
            layout = Table([[image, side]], colWidths=[left, width - left])
            layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                        ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
            return [KeepTogether(parts + [layout])]
        if images and result.image:
            w_px, h_px = result.size_px or (1200, 800)
            w_pt, h_pt = w_px / DPI * 72, h_px / DPI * 72
            scale = min(width / w_pt, (height * 0.66) / h_pt, 1.0)
            parts += [Spacer(1, 4), Image(io.BytesIO(result.image), width=w_pt * scale, height=h_pt * scale,
                                          hAlign="CENTER")]
            self.images_embedded += 1
        elif not images:
            parts.append(Paragraph("<i>Figure omitted: the PDF was rebuilt without figures (see appendix).</i>",
                                   styles["caption"]))
        parts += caption
        block = [KeepTogether(parts)]
        if result.metrics:
            block.append(self._kv_table(list(result.metrics.items()), styles, min(width, 560)))
        return block

    def _index(self, viz: VisualizationReport, styles, width: float) -> list:
        from reportlab.platypus import Paragraph, Spacer, Table
        intro = Paragraph(_text(index_intro(viz)), styles["body"])
        data = [[Paragraph(h, styles["cellhead"]) for h in INDEX_HEADER]]
        for i, r, detail in index_rows(viz):
            status = Paragraph(f'<font color="{STATUS_INK[r.status]}"><b>{r.status.value.upper()}</b></font>',
                               styles["cell"])
            data.append([Paragraph(str(i), styles["cell"]), Paragraph(_text(r.title), styles["cell"]),
                         Paragraph(_text(r.section), styles["cell"]), status,
                         Paragraph(f"{r.seconds:.2f}", styles["cell"]), Paragraph(_text(detail), styles["cell"])])
        fractions = [0.04, 0.26, 0.2, 0.08, 0.07, 0.35]
        table = Table(data, colWidths=[width * f for f in fractions], repeatRows=1, hAlign="LEFT")
        table.setStyle(self._table_style(header=True))
        return [intro, Spacer(1, 6), table]

    def _problems(self, viz: VisualizationReport, styles, images: bool) -> list:
        from reportlab.platypus import Paragraph, Preformatted, Spacer
        out = []
        if self.warnings:
            out.append(Paragraph("Report warnings", styles["h2"]))
            out += [Paragraph(_text(w), styles["body"]) for w in self.warnings]
        if viz.failed:
            out.append(Paragraph(f"Failed visualizations ({len(viz.failed)})", styles["h2"]))
            out.append(Paragraph(_text(FAILED_NOTE), styles["body"]))
            for r in viz.failed:
                out.append(Paragraph(f"<b>{_text(r.title)}</b> ({_text(r.key)}) - {_text(r.error_type)}: "
                                     f"{_text(r.reason)}", styles["body"]))
                tail = "\n".join(r.traceback.strip().splitlines()[-6:])
                if tail:
                    out.append(Preformatted(tail, styles["pre"]))
                out.append(Spacer(1, 4))
        if viz.skipped:
            out.append(Paragraph(f"Skipped visualizations ({len(viz.skipped)})", styles["h2"]))
            out.append(Paragraph(_text(SKIPPED_NOTE), styles["body"]))
            out += [Paragraph(f"<b>{_text(r.title)}</b>: {_text(r.reason)}", styles["body"]) for r in viz.skipped]
        return out
