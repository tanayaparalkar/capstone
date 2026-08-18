"""
Markdown report generator.

Builds the professional Markdown security report from an existing
ScanResult and ScanStatistics - like the JSON report, this does not
recalculate statistics (calculate_statistics() remains the single source
of truth) and does not redefine finding fields (ScannerFinding /
AIEnrichedFinding remain the only source of truth for their own data).
No second data model is introduced for Markdown; this module only
renders text from the existing contracts + statistics.

Findings are sorted by finding_id everywhere in this report - the same
stable ordering the JSON report uses - so the two formats present
findings identically and neither introduces randomness.

Security note: raw_evidence and patch_suggestion are scanner/LLM-sourced
text that could contain backticks, headings, or HTML. Both are rendered
inside a fenced code block whose fence is dynamically sized longer than
any backtick run already in the content (see _fenced_code), so nothing
in that text can break out of the code block or be interpreted as
Markdown/HTML structure.
"""
from datetime import datetime
from typing import Optional

from ..contracts import AIEnrichedFinding, ScanResult, ScannerFinding
from ..core import build_ai_lookup, build_correlation_lookup, format_location
from ..statistics import ScanStatistics


def to_markdown(result: ScanResult, statistics: ScanStatistics) -> str:
    scanner_findings = sorted(result.scanner_findings, key=lambda f: f.finding_id)
    ai_by_id = build_ai_lookup(result)
    corr_by_id = build_correlation_lookup(result)

    sections = [
        "# SentinelAI Security Report",
        _executive_summary(result, statistics),
        _scan_information(result, statistics),
    ]

    if scanner_findings:
        sections.append(_findings_overview(scanner_findings, ai_by_id))
        sections.append(_detailed_findings(scanner_findings, ai_by_id, corr_by_id))
    else:
        sections.append("## Findings\n\nNo findings were reported for this scan.")

    return "\n\n".join(sections) + "\n"


def _executive_summary(result: ScanResult, stats: ScanStatistics) -> str:
    lines = [
        "## Executive Summary",
        "",
        f"- **Repository:** {result.repository.name}",
        f"- **Scan Mode:** {stats.scan_mode}",
        f"- **Scanner Tier:** {result.metadata.scanner_tier.value}",
        f"- **Scan Timestamp:** {_iso(stats.timestamp)}",
        f"- **Scan Duration:** {_duration(stats.duration_seconds)}",
        f"- **Total Findings:** {stats.total_findings}",
        f"- **Correlated Issues:** {stats.correlated_findings}"
        + (f" ({stats.multi_scanner_findings} confirmed by 2+ scanners)" if stats.multi_scanner_findings else ""),
        f"- **AI Enrichment:** {stats.ai_enrichment_status.value}",
        "",
        "| Severity | Count |",
        "|---|---:|",
        f"| Critical | {stats.critical_findings} |",
        f"| High | {stats.high_findings} |",
        f"| Medium | {stats.medium_findings} |",
        f"| Low | {stats.low_findings} |",
    ]

    if stats.scanner_counts:
        lines += ["", "**Findings by Scanner**", "", "| Scanner | Count |", "|---|---:|"]
        lines += [f"| {name} | {count} |" for name, count in sorted(stats.scanner_counts.items())]

    if stats.category_counts:
        lines += ["", "**Findings by Category**", "", "| Category | Count |", "|---|---:|"]
        lines += [f"| {name} | {count} |" for name, count in sorted(stats.category_counts.items())]

    return "\n".join(lines)


def _scan_information(result: ScanResult, stats: ScanStatistics) -> str:
    repo = result.repository
    lines = [
        "## Scan Information",
        "",
        f"- **Repository Name:** {repo.name}",
        f"- **Repository Path:** {repo.path}",
        f"- **Scan Mode:** {stats.scan_mode}",
        f"- **Scanner Tier:** {result.metadata.scanner_tier.value}",
        f"- **Timestamp:** {_iso(stats.timestamp)}",
        f"- **Duration:** {_duration(stats.duration_seconds)}",
    ]
    if repo.commit_hash:
        lines.append(f"- **Commit:** {repo.commit_hash}")
    if repo.branch:
        lines.append(f"- **Branch:** {repo.branch}")
    if repo.languages:
        lines.append(f"- **Languages:** {', '.join(repo.languages)}")
    return "\n".join(lines)


def _findings_overview(scanner_findings: list[ScannerFinding], ai_by_id: dict[str, AIEnrichedFinding]) -> str:
    lines = [
        "## Findings Overview",
        "",
        "| ID | Severity | Scanner | Category | Location | Rule | Confidence | Verification |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in scanner_findings:
        ai = ai_by_id.get(f.finding_id)
        confidence = ai.confidence_label.value if ai else "pending"
        verification = ai.verification_status.value if ai else "—"
        lines.append(
            f"| {f.finding_id} | {f.severity.value.upper()} | {f.scanner} | {f.category} "
            f"| {format_location(f)} | `{f.rule_id}` | {confidence} | {verification} |"
        )
    return "\n".join(lines)


def _detailed_findings(scanner_findings, ai_by_id, corr_by_id) -> str:
    sections = ["## Detailed Findings"]
    for f in scanner_findings:
        sections.append(_finding_section(f, ai_by_id.get(f.finding_id), corr_by_id.get(f.finding_id)))
    return "\n\n".join(sections)


def _finding_section(f: ScannerFinding, ai: Optional[AIEnrichedFinding], corr=None) -> str:
    lines = [
        f"### {f.finding_id} — {f.category} ({f.severity.value.upper()})",
        "",
        f"- **Scanner:** {f.scanner}",
        f"- **Rule:** `{f.rule_id}`",
        f"- **Location:** {format_location(f)}",
    ]
    if f.cwe:
        lines.append(f"- **CWE:** {f.cwe}")
    # Make the canonical/source relationship visible: a reader must be able to see
    # that this raw finding was grouped, which other raw findings it was grouped
    # with, and on what basis.
    if corr is not None and len(corr.source_finding_ids) > 1:
        others = [i for i in corr.source_finding_ids if i != f.finding_id]
        role = "canonical" if corr.canonical_finding_id == f.finding_id else "grouped under " + corr.canonical_finding_id
        lines.append(f"- **Correlated Issue:** {corr.correlation_id} ({role})")
        lines.append(f"- **Grouped With:** {', '.join(others)}")
        lines.append(f"- **Correlated Scanners:** {', '.join(corr.scanners)}")
        lines.append(f"- **Correlation Basis:** {corr.correlation_reason}")

    lines += ["", "**Description**", "", f.message]

    if f.raw_evidence:
        lines += ["", "**Evidence**", "", _fenced_code(f.raw_evidence)]

    if ai is not None:
        lines += ["", "**AI Explanation**", "", ai.explanation]
        if ai.exploit_path:
            lines += ["", "**Exploit Path**", "", ai.exploit_path]
        if ai.impact:
            lines += ["", "**Impact**", "", ai.impact]
        lines += ["", "**Remediation**", "", ai.remediation]
        if ai.patch_suggestion:
            lines += ["", "**Patch Suggestion**", "", _fenced_code(ai.patch_suggestion)]
        lines += [
            "",
            f"- **Confidence:** {ai.confidence_label.value.upper()} ({ai.confidence_score:.2f})",
            f"- **Verification:** {ai.verification_status.value}",
        ]
        # Critic review signals, rendered separately from Verification above:
        # verification_status is the deterministic category check, grounding_verdict
        # is the critic agent's judgement. Omitted entirely when no critic ran, so
        # reports predating the multi-agent pipeline render exactly as before.
        if ai.grounding_verdict is not None:
            lines.append(f"- **Grounding (critic):** {ai.grounding_verdict.value}")
        if ai.unsupported_claims:
            lines += ["", "**Unsupported or Uncertain Claims**", ""]
            lines += [f"- {claim}" for claim in ai.unsupported_claims]
        if ai.related_findings:
            lines.append(f"- **Related Findings:** {', '.join(ai.related_findings)}")
        if ai.references:
            lines.append(f"- **References:** {', '.join(ai.references)}")
    else:
        lines += ["", "_AI enrichment not yet available for this finding._"]

    return "\n".join(lines)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _duration(value: Optional[float]) -> str:
    return f"{value:.2f}s" if value is not None else "unavailable"


def _fenced_code(content: str) -> str:
    """Wrap content in a code fence sized longer than any backtick run already
    inside it, so backticks/headings/HTML in the content can't break the
    document structure or be interpreted as anything but literal text."""
    longest_run = 0
    run = 0
    for ch in content:
        if ch == "`":
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}\n{content}\n{fence}"
