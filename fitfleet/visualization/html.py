"""HTML reports: the same content as the PDFs, as one self-contained file.

`render_report()` turns a PdfReport into HTML using the same sections, numbering, text, tables,
figures, explanations and appendix as `PdfReport.build()`. Both formats read
`PdfReport.sections()` and the helpers in fitfleet.visualization.pdf, so they can't drift apart.
Figures are embedded as base64 PNGs, so the file can be opened, mailed or archived on its own;
nothing is loaded from the network. When printed, sections start on a new page wherever the
PDF starts one.

`document()` and `image_tag()` are also used by reports that are not built with a PdfReport
(FeatureResearch).
"""
from __future__ import annotations

import base64
import html
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from fitfleet.visualization.core import DPI, VisualizationReport
from fitfleet.visualization.pdf import (ACCENT, APPENDIX, FAILED_NOTE, INK, MUTED, RULE, SKIPPED_NOTE, STATUS_INK,
                                        ZEBRA, INDEX_HEADER, figure_meta, frame_rows, index_intro, index_rows)

if TYPE_CHECKING:
    from fitfleet.visualization.pdf import PdfReport, Section

CSS_PX_PER_INCH = 96


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


STYLE = f"""
*{{box-sizing:border-box}}
body{{margin:0;background:#f3f3f1;color:{INK};font:14px/1.5 Helvetica,Arial,"Segoe UI",sans-serif}}
.page{{max-width:var(--width);margin:24px auto;background:#fff;padding:36px 40px 48px;
  box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.running{{display:flex;justify-content:space-between;font-size:11px;color:{MUTED};
  border-bottom:.5px solid {RULE};padding-bottom:4px;margin-bottom:18px}}
h1.title{{font-size:27px;line-height:1.2;margin:0 0 4px}}
.subtitle{{color:{MUTED};font-size:13.5px;margin:0 0 14px}}
nav.toc{{font-size:12.5px;margin:0 0 8px;padding:10px 14px;background:{ZEBRA};border-radius:4px}}
nav.toc ol{{margin:0;padding-left:20px;columns:2;column-gap:28px}}
nav.toc a{{color:{ACCENT};text-decoration:none}}
nav.toc a:hover{{text-decoration:underline}}
section{{margin-top:28px}}
section>h2.section{{font-size:20px;color:{ACCENT};margin:0 0 10px;padding-top:6px;border-top:1px solid {RULE}}}
h3{{font-size:15.5px;margin:14px 0 5px}}
p{{margin:0 0 8px}}
ul{{margin:0 0 8px;padding-left:22px}}
pre{{font:11.5px/1.35 Consolas,"Cascadia Code",Menlo,monospace;background:{ZEBRA};padding:8px 10px;
  overflow-x:auto;white-space:pre;margin:0 0 8px;border-radius:3px}}
.scroll{{overflow-x:auto;margin:0 0 10px}}
table{{border-collapse:collapse;font-size:12px;margin:0 0 10px}}
th,td{{text-align:left;vertical-align:top;padding:3px 6px;border-bottom:.5px solid {RULE}}}
table.data{{width:100%}}
table.data thead th{{border-bottom:1.2px solid {MUTED};position:sticky;top:0;background:#fff}}
table.data tbody tr:nth-child(even){{background:{ZEBRA}}}
table.kv{{max-width:720px;width:100%}}
table.kv th{{width:36%;font-weight:bold}}
figure{{margin:18px 0 6px}}
figure h3{{margin:0 0 2px}}
.meta{{font-size:10.5px;color:{MUTED};margin:0 0 6px}}
.caption{{font-size:12px;color:{MUTED};margin:4px 0}}
figure img{{display:block;max-width:100%;height:auto;margin:6px auto}}
.side{{display:flex;gap:2%;align-items:flex-start}}
.side>.fig{{flex:0 0 64%}}
.side>.fig img{{margin:4px 0}}
.side>.text{{flex:1 1 34%;min-width:0}}
.status{{font-weight:bold}}
pre.line{{background:none;margin:0;padding:0 2px;overflow:visible}}
.page-break{{height:0}}
footer{{margin-top:36px;font-size:11px;color:{MUTED};border-top:.5px solid {RULE};padding-top:6px}}
@media (max-width:820px){{.page{{padding:20px 16px;margin:0}} .side{{display:block}} nav.toc ol{{columns:1}}}}
@media print{{
  body{{background:#fff}} .page{{box-shadow:none;margin:0;max-width:none;padding:0}}
  nav.toc{{display:none}} section.new-page,.page-break{{break-before:page}} figure{{break-inside:avoid}}
  table.data thead th{{position:static}} a{{color:inherit}}
}}
"""


def document(title: str, body: str, *, subtitle: str = "", created: str = "", landscape: bool = True,
             contents: Iterable[tuple[str, str]] = ()) -> str:
    """A complete HTML page: running header (title, created), title, subtitle, contents, `body`.

    `contents`: (anchor id, label) pairs for the table of contents. `landscape` makes the page as
    wide as a landscape A4 PDF page (and prints in landscape)."""
    width = "1180px" if landscape else "860px"
    toc = "".join(f'<li><a href="#{_e(anchor)}">{_e(label)}</a></li>' for anchor, label in contents)
    return "\n".join([
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="generator" content="fitfleet">',
        f"<title>{_e(title)}</title>",
        f"<style>:root{{--width:{width}}}\n@page{{size:A4 {'landscape' if landscape else 'portrait'};"
        f"margin:14mm 12mm}}{STYLE}</style>",
        "</head>",
        "<body>",
        '<main class="page">',
        f'<div class="running"><span>{_e(title)}</span><span>{_e(created)}</span></div>',
        f'<h1 class="title">{_e(title)}</h1>',
        f'<p class="subtitle">{_e(subtitle)}</p>' if subtitle else "",
        f'<nav class="toc" aria-label="Contents"><ol>{toc}</ol></nav>' if toc else "",
        body,
        f"<footer>generated by fitfleet{' · ' + _e(created) if created else ''}</footer>",
        "</main>",
        "</body>",
        "</html>",
    ])


def image_tag(png: bytes, alt: str, width_px: float | None = None) -> str:
    """An <img> with the PNG embedded; `width_px` is its natural CSS width (it still shrinks to fit)."""
    data = base64.b64encode(png).decode("ascii")
    width = f' style="width:{width_px:.0f}px"' if width_px else ""
    return f'<img src="data:image/png;base64,{data}" alt="{_e(alt)}"{width}>'


def kv_table(rows: Iterable[tuple[Any, Any]]) -> str:
    """Key / value rows as a two-column table (like the PDF's key-value tables)."""
    body = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in rows)
    return f'<table class="kv"><tbody>{body}</tbody></table>'


def data_table(header: Iterable[Any], rows: Iterable[Iterable[Any]], cell_html: bool = False) -> str:
    """A table with a header row. `cell_html`: the cells are already HTML (otherwise escaped)."""
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c if cell_html else _e(c)}</td>" for c in row) + "</tr>"
                   for row in rows)
    return f'<div class="scroll"><table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def write_atomic(path: Path, text: str) -> Path:
    """Write `text` to `path` via a temporary file, so a failure never leaves a truncated report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, path)
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
    return path


# ---------------------------------------------------------------------------- PdfReport -> HTML
def render_report(report: PdfReport, viz: VisualizationReport, identifiers: dict[str, str]) -> str:
    """The HTML page of `report`, section for section the same as its PDF."""
    sections = report.sections(viz, images=True)
    body = [_section(report, s, viz, identifiers) for s in sections]
    return document(report.title, "\n".join(body), subtitle=report.subtitle,
                    created=f"{report.created:%Y-%m-%d %H:%M}", landscape=report.landscape,
                    contents=[(f"section-{s.number}", f"{s.number}. {s.name}") for s in sections])


def _section(report: PdfReport, section: Section, viz: VisualizationReport, identifiers: dict[str, str]) -> str:
    cls = ' class="new-page"' if section.new_page else ""
    out = [f'<section id="section-{section.number}"{cls}>',
           f'<h2 class="section">{section.number}. {_e(section.name)}</h2>']
    out += [_block(kind, payload) for kind, payload in section.blocks]
    out += [_figure(report, result, identifiers) for result in section.figures]
    if section.name == APPENDIX[0]:
        out.append(_index(viz))
    elif section.name == APPENDIX[1]:
        out.append(_problems(report, viz))
    out.append("</section>")
    return "\n".join(out)


def _block(kind: str, payload: Any) -> str:
    if kind == "paragraph":
        return f"<p>{_e(payload)}</p>"
    if kind == "heading":
        return f"<h3>{_e(payload)}</h3>"
    if kind == "bullets":
        return "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in payload) + "</ul>"
    if kind == "kv":
        return kv_table(payload)
    if kind == "table":
        frame, index, digits = payload
        return data_table(*frame_rows(frame, index, digits))
    if kind == "pre":
        return f"<pre>{_e(payload)}</pre>"
    raise ValueError(kind)


def _figure(report: PdfReport, result, identifiers: dict[str, str]) -> str:
    head = [f"<h3>{_e(result.title)}</h3>", f'<p class="meta">{_e(figure_meta(result, identifiers))}</p>']
    caption = [f'<p class="caption"><b>What it shows.</b> {_e(result.description)}</p>']
    if result.note:
        caption.append(f'<p class="caption">{_e(result.note)}</p>')
    values = kv_table(result.metrics.items()) if result.metrics else ""
    if not result.image:
        image = '<p class="caption"><i>Figure omitted: no image was recorded for this plot.</i></p>'
        return "\n".join(["<figure>", *head, image, *caption, values, "</figure>"])
    w_px, _ = result.size_px or (1200, 800)
    image = image_tag(result.image, result.title, w_px / DPI * CSS_PX_PER_INCH)
    if report.side_by_side(result):  # as in the landscape PDF: figure left, explanation and values right
        return "\n".join(["<figure>", *head, '<div class="side">', f'<div class="fig">{image}</div>',
                          '<div class="text">', *caption, values, "</div>", "</div>", "</figure>"])
    return "\n".join(["<figure>", *head, image, *caption, "</figure>", values])


def _index(viz: VisualizationReport) -> str:
    rows = []
    for i, r, detail in index_rows(viz):
        status = f'<span class="status" style="color:{STATUS_INK[r.status]}">{_e(r.status.value.upper())}</span>'
        rows.append([str(i), _e(r.title), _e(r.section), status, f"{r.seconds:.2f}", _e(detail)])
    return f"<p>{_e(index_intro(viz))}</p>" + data_table(INDEX_HEADER, rows, cell_html=True)


def _problems(report: PdfReport, viz: VisualizationReport) -> str:
    out = []
    if report.warnings:
        out.append("<h3>Report warnings</h3>")
        out += [f"<p>{_e(w)}</p>" for w in report.warnings]
    if viz.failed:
        out.append(f"<h3>Failed visualizations ({len(viz.failed)})</h3>")
        out.append(f"<p>{_e(FAILED_NOTE)}</p>")
        for r in viz.failed:
            out.append(f"<p><b>{_e(r.title)}</b> ({_e(r.key)}) - {_e(r.error_type)}: {_e(r.reason)}</p>")
            tail = "\n".join(r.traceback.strip().splitlines()[-6:])
            if tail:
                out.append(f"<pre>{_e(tail)}</pre>")
    if viz.skipped:
        out.append(f"<h3>Skipped visualizations ({len(viz.skipped)})</h3>")
        out.append(f"<p>{_e(SKIPPED_NOTE)}</p>")
        out += [f"<p><b>{_e(r.title)}</b>: {_e(r.reason)}</p>" for r in viz.skipped]
    return "\n".join(out)
