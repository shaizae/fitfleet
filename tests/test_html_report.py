"""HTML reports: the same content as the PDFs, in one self-contained file."""
import html
import re

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from fitfleet import Classification, FeatureResearch, FilteringCriteriaClassification, Regression, ReportFormat
from fitfleet.config import ConfigError, Experiment
from fitfleet.results.experiment import ExperimentReport, _add_execution
from fitfleet.visualization import VisualizationManager
from fitfleet.visualization.core import VisualizationReport, VisualizationResult, VizStatus
from fitfleet.visualization.pdf import PdfReport


@pytest.fixture(scope="module")
def iris_run():
    d = load_iris()
    X, y = pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target_names[d.target])
    with Classification() as run:
        run.random_state = 0
        run.set(X, y, [LogisticRegression(max_iter=1000), DecisionTreeClassifier(random_state=0)])
        run.set_hyper_parameters({"max_depth": [2, 3]})
        run.k_folds(3)
        best = run.filter_by(FilteringCriteriaClassification.accuracy)
        yield run, run.experiment_report(), best[0]


def _images(page: str) -> int:
    return len(re.findall(r'<img src="data:image/png;base64,', page))


def _section_titles(page: str) -> list[str]:
    return re.findall(r'<h2 class="section">(\d+\. [^<]+)</h2>', page)


# ------------------------------------------------------------------ formats
@pytest.mark.parametrize("value, expected", [(ReportFormat.pdf, ("pdf",)), (ReportFormat.html, ("html",)),
                                             (ReportFormat.dual, ("pdf", "html")), ("HTML", ("html",)),
                                             ("dual", ("pdf", "html"))])
def test_report_format(value, expected):
    assert ReportFormat.parse(value).files == expected


def test_report_format_labels():
    assert [f.label for f in ReportFormat] == ["PDF", "HTML", "PDF + HTML"]


@pytest.mark.parametrize("value", ["docx", "", "both", None, ["pdf"]])
def test_report_format_rejects_unknown(value):
    with pytest.raises(ValueError, match="report format"):
        ReportFormat.parse(value)


def test_save_report_rejects_an_unknown_format(tmp_path, iris_run):
    _, report, best = iris_run
    with pytest.raises(ValueError, match="report format"):
        report.save_report(tmp_path, "docx")
    with pytest.raises(ValueError, match="report format"):
        best.save_report(str(tmp_path), "docx")
    assert not list(tmp_path.iterdir())


# ------------------------------------------------------------------ same content as the PDF
def test_experiment_html_has_every_section_and_plot_of_the_pdf(tmp_path, iris_run):
    _, report, _ = iris_run
    (path,) = report.save_report(tmp_path, ReportFormat.html)
    assert path.name.startswith("experiment_report_") and path.suffix == ".html"
    page = path.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "</html>" in page

    viz = report.visualize()  # rebuild the PdfReport the same way save_report does, and compare
    pdf = report._pdf(viz, 20)
    _add_execution(pdf, report._execution)
    expected = [f"{s.number}. {s.name}" for s in pdf.sections(viz)]
    assert _section_titles(page) == expected
    assert _images(page) == len(viz.succeeded) > 5
    for result in viz.succeeded:  # every plot keeps its title and explanation
        assert f"<h3>{html.escape(result.title)}</h3>" in page
        assert html.escape(result.description) in page
    assert re.search(r"<h3>Top \d+ configurations</h3>", page)
    assert not re.search(r'(src|href)="https?://', page)  # self-contained: nothing loaded from the network


def test_save_report_both_draws_the_plots_once(tmp_path, iris_run, monkeypatch):
    _, report, _ = iris_run
    calls = []
    original = ExperimentReport.visualize

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(ExperimentReport, "visualize", counting)
    pdf_path, html_path = report.save_report(tmp_path, ReportFormat.dual)
    assert len(calls) == 1
    assert pdf_path.suffix == ".pdf" and html_path.suffix == ".html" and pdf_path.stem == html_path.stem
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert not list(tmp_path.glob("*.part"))


def test_per_model_html_report_classification(tmp_path, iris_run):
    _, _, best = iris_run
    (path,) = best.save_report(str(tmp_path), ReportFormat.html)
    page = path.read_text(encoding="utf-8")
    assert path.name.startswith(f"report_{best.name}_") and path.suffix == ".html"
    assert f"Classification Report - {best.name}" in page
    assert "1. Executive Summary" in page and "Additional metrics" in page
    assert _images(page) >= 3


def test_per_model_html_report_regression(tmp_path):
    d = load_diabetes()
    with Regression() as run:
        run.random_state = 0
        run.set(pd.DataFrame(d.data, columns=d.feature_names), d.target, [Ridge()])
        run.k_folds(3)
        best = run.filter_by(run.default_criteria)[0]
    pdf_path, html_path = best.save_report(str(tmp_path), "dual")
    page = html_path.read_text(encoding="utf-8")
    assert pdf_path.suffix == ".pdf" and f"Regression Report - {best.name}" in page
    assert "Cross-Validation Analysis" in page and _images(page) >= 3


def test_save_results_can_write_both_formats(tmp_path, iris_run):
    _, _, best = iris_run
    best.save_results(str(tmp_path), ReportFormat.dual)
    (folder,) = tmp_path.iterdir()
    assert sorted(p.suffix for p in folder.iterdir()) == [".html", ".joblib", ".pdf"]


def test_failed_and_skipped_plots_are_listed_like_in_the_pdf(tmp_path):
    pdf = PdfReport("T <script>", "sub & title")
    pdf.paragraph("Executive Summary", "a < b")
    viz = VisualizationReport([
        VisualizationResult("ok", "Good plot", "Overall Results and Model Comparison", "shows it",
                            VizStatus.SUCCESS, image=b"\x89PNG", size_px=(600, 400), metrics={"auc": 0.9}),
        VisualizationResult("bad", "Broken plot", "Overall Results and Model Comparison", "d", VizStatus.FAILED,
                            reason="boom", error_type="RuntimeError", traceback="Traceback\n  line\nRuntimeError"),
        VisualizationResult("na", "Missing plot", "Overall Results and Model Comparison", "d", VizStatus.SKIPPED,
                            reason="needs probabilities")])
    page = pdf.build_html(tmp_path / "r.html", viz).read_text(encoding="utf-8")
    assert "T &lt;script&gt;" in page and "<script>" not in page and "a &lt; b" in page
    titles = _section_titles(page)
    assert titles == [f"{s.number}. {s.name}" for s in pdf.sections(viz)]
    assert titles[-2:] == ["3. Visualization Index", "4. Errors, Warnings and Missing Visualizations"]
    assert "Failed visualizations (1)" in page and "RuntimeError: boom" in page
    assert "Skipped visualizations (1)" in page and "needs probabilities" in page
    assert _images(page) == 1


def test_a_pdf_warning_does_not_reach_the_html(tmp_path, monkeypatch):
    pdf = PdfReport("T")
    pdf.paragraph("Executive Summary", "x")
    original = PdfReport._build

    def figures_fail_once(self, path, viz, identifiers, images):
        if images:
            raise OSError("image engine down")
        return original(self, path, viz, identifiers, images)

    monkeypatch.setattr(PdfReport, "_build", figures_fail_once)
    pdf_path, html_path = pdf.write(tmp_path / "r", ReportFormat.dual)
    assert "image engine down" not in html_path.read_text(encoding="utf-8")
    assert pdf_path.exists()


# ------------------------------------------------------------------ execution log
def test_save_html_log(tmp_path, iris_data):
    with Classification(log_dir=tmp_path) as run:
        run.set(*iris_data, [DecisionTreeClassifier(random_state=0)])
        run.k_folds(3)
    path = run.save_html_log
    page = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path and path.name.startswith("execution-log_") and path.suffix == ".html"
    assert "Execution Log" in page and "1. Execution Summary" in page and "Stage tree" in page


# ------------------------------------------------------------------ feature research
def test_feature_research_html_matches_its_pdf(tmp_path):
    d = load_iris()
    research = FeatureResearch(pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target),
                               random_state=0, cv=3)
    pdf_path, html_path = research.save_report(tmp_path, ReportFormat.dual, probe=False)
    assert pdf_path.suffix == ".pdf" and html_path.suffix == ".html" and pdf_path.stem == html_path.stem
    page = html_path.read_text(encoding="utf-8")
    for heading in ("Recommended to drop", "Possible leakage", "Feature ranking", "Data quality", "Correlation"):
        assert f">{heading}</h2>" in page
    assert _images(page) == 2  # importance chart and correlation heatmap, as in the PDF
    assert [p.suffix for p in research.save_report(tmp_path, ReportFormat.html, probe=False)] == [".html"]


# ------------------------------------------------------------------ fitfleet run
def _config(tmp_path, **output):
    d = load_iris()
    frame = pd.DataFrame(d.data, columns=["a", "b", "c", "d"])
    frame["label"] = d.target_names[d.target]
    frame.to_csv(tmp_path / "data.csv", index=False)
    return Experiment.from_dict({
        "data": {"path": "data.csv", "target": "label", "task": "classification"},
        "run": {"validation": "k_folds", "n_splits": 3, "random_state": 0},
        "models": [{"name": "DecisionTreeClassifier", "params": {"random_state": 0}}],
        "output": {"dir": "runs", "keep": 1, **output}}, base=tmp_path)


def test_config_report_format_dual(tmp_path):
    lines = []
    experiment = _config(tmp_path, report_format="dual")
    folder = experiment.run(log=lines.append)
    assert len(list(folder.glob("experiment_report_*.pdf"))) == 1
    assert len(list(folder.glob("experiment_report_*.html"))) == 1
    (place,) = folder.glob("1_*")
    assert {p.suffix for p in place.glob("report_*")} == {".pdf", ".html"}
    assert sum(line.startswith("report -> ") for line in lines) == 4
    log = experiment.last_execution.log()
    assert log.find("Experiment PDF + HTML report") is not None
    assert log.find("Finish PDF + HTML reports") is not None


def test_config_report_format_html_only(tmp_path, monkeypatch):
    monkeypatch.setattr(PdfReport, "build", lambda *a, **k: pytest.fail("a PDF was built"))
    folder = _config(tmp_path, report_format="html").run(log=lambda s: None)
    assert list(folder.glob("experiment_report_*.html")) and not list(folder.rglob("*.pdf"))


def test_config_report_format_is_validated(tmp_path):
    with pytest.raises(ConfigError, match="report_format"):
        _config(tmp_path, report_format="docx")


def test_config_without_reports_renders_no_html(tmp_path, monkeypatch):
    monkeypatch.setattr(VisualizationManager, "run", lambda *a: pytest.fail("rendered with report = false"))
    folder = _config(tmp_path, report=False, report_format="dual").run(log=lambda s: None)
    assert not list(folder.rglob("*.html"))
