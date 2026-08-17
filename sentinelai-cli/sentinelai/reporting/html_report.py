"""
HTML report generator.

Renders the professional, self-contained static HTML security report
from an existing ScanResult and ScanStatistics via a Jinja2 template.
Like the JSON and Markdown generators, this does not recalculate
statistics (calculate_statistics() remains the single source of truth)
or redefine finding fields (ScannerFinding / AIEnrichedFinding remain the
only source of truth) - it only prepares data for the template: scanner
<-> AI correlation by finding_id, stable finding_id ordering, display
percentages for the CSS bar visualizations, and safe anchor slugs. The
template (templates/security_report.html.j2) is responsible purely for
presentation.

Security: the Jinja2 Environment has autoescape permanently enabled and
no template output uses the `|safe` filter - every piece of finding/AI
content (which may eventually include LLM-generated text or embedded
code snippets) is HTML-escaped automatically, so it can never execute as
markup in the rendered report.
"""
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..contracts import ScanResult
from ..core import build_ai_lookup, build_correlation_lookup, format_location
from ..statistics import ScanStatistics

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=True)


def to_html(result: ScanResult, statistics: ScanStatistics) -> str:
    template = _env.get_template("security_report.html.j2")
    return template.render(**_build_context(result, statistics))


def _build_context(result: ScanResult, stats: ScanStatistics) -> dict:
    scanner_findings = sorted(result.scanner_findings, key=lambda f: f.finding_id)
    ai_by_id = build_ai_lookup(result)
    corr_by_id = build_correlation_lookup(result)
    known_anchors = {f.finding_id: _anchor(f.finding_id) for f in scanner_findings}

    findings = []
    for f in scanner_findings:
        ai = ai_by_id.get(f.finding_id)
        corr = corr_by_id.get(f.finding_id)
        findings.append(
            {
                "finding": f,
                "ai": ai,
                "anchor": known_anchors[f.finding_id],
                "location": format_location(f),
                "ai_confidence_percent": round(ai.confidence_score * 100) if ai else None,
                "correlation": corr if (corr is not None and len(corr.source_finding_ids) > 1) else None,
                "correlation_others": (
                    [i for i in corr.source_finding_ids if i != f.finding_id]
                    if corr is not None and len(corr.source_finding_ids) > 1
                    else []
                ),
                "is_canonical": corr is not None and corr.canonical_finding_id == f.finding_id,
                "related": [
                    {"id": rid, "anchor": known_anchors.get(rid)} for rid in (ai.related_findings if ai else [])
                ],
                "references": [_reference(ref) for ref in (ai.references if ai else [])],
            }
        )

    return {
        "repository": result.repository,
        "stats": stats,
        "findings": findings,
        "severity_bars": _severity_bars(stats),
        "scanner_bars": _distribution_bars(stats.scanner_counts, stats.total_findings),
        "category_bars": _distribution_bars(stats.category_counts, stats.total_findings),
    }


def _severity_bars(stats: ScanStatistics) -> list[dict]:
    counts = [
        ("Critical", "critical", stats.critical_findings),
        ("High", "high", stats.high_findings),
        ("Medium", "medium", stats.medium_findings),
        ("Low", "low", stats.low_findings),
    ]
    return [
        {"label": label, "key": key, "count": count, "percent": _percent(count, stats.total_findings)}
        for label, key, count in counts
        if count > 0
    ]


def _distribution_bars(counts: dict, total: int) -> list[dict]:
    return [{"label": name, "count": count, "percent": _percent(count, total)} for name, count in sorted(counts.items())]


def _percent(count: int, total: int) -> float:
    return round(count / total * 100, 1) if total > 0 else 0.0


def _anchor(finding_id: str) -> str:
    """Normalize a finding_id into a safe HTML id fragment (finding-<anchor>)."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", finding_id)


def _reference(ref: str) -> dict:
    is_url = ref.startswith("http://") or ref.startswith("https://")
    return {"text": ref, "url": ref if is_url else None}
