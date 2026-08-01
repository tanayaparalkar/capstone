"""
Non-interactive output formatters: JSON, Markdown, HTML.

These are intentionally simple in Phase 1 - just enough structure to be
genuinely useful. Phase 3 will likely replace the markdown/html bodies
with more polished, styled templates, but the function signatures here
are the contract the CLI (and later the web frontend) can keep calling.
"""
import json
from datetime import datetime, timezone

from .models import Finding

SEVERITY_ORDER = ["critical", "high", "medium", "low"]


def to_json(findings: list[Finding]) -> str:
    return json.dumps([f.model_dump() for f in findings], indent=2)


def to_markdown(findings: list[Finding], repo_path: str) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# SentinelAI Vulnerability Report",
        "",
        f"- **Repository:** `{repo_path}`",
        f"- **Generated:** {generated}",
        f"- **Total findings:** {len(findings)}",
        "",
    ]

    grouped: dict[str, list[Finding]] = {sev: [] for sev in SEVERITY_ORDER}
    for f in findings:
        grouped[f.severity.value].append(f)

    for sev in SEVERITY_ORDER:
        group = grouped[sev]
        if not group:
            continue
        lines.append(f"## {sev.upper()} ({len(group)})")
        lines.append("")
        for f in group:
            lines.append(f"### `{f.file}:{f.line}` \u2014 {f.type} ({f.id})")
            lines.append(f"**Rule:** `{f.rule}`  ")
            lines.append(f"**Confidence:** {f.confidence.value}")
            lines.append("")
            lines.append(f.ai_description)
            lines.append("")
            if f.exploit_path:
                lines.append(f"**Exploit path:** {f.exploit_path}")
                lines.append("")
            lines.append(f"**Remediation:** {f.remediation}")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def to_html(findings: list[Finding], repo_path: str) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    severity_colors = {
        "critical": "#dc2626",
        "high": "#ea580c",
        "medium": "#ca8a04",
        "low": "#2563eb",
    }

    grouped: dict[str, list[Finding]] = {sev: [] for sev in SEVERITY_ORDER}
    for f in findings:
        grouped[f.severity.value].append(f)

    rows = []
    for sev in SEVERITY_ORDER:
        for f in grouped[sev]:
            exploit_html = (
                f"<p><strong>Exploit path:</strong> {f.exploit_path}</p>"
                if f.exploit_path
                else ""
            )
            rows.append(f"""
      <div class="finding" style="border-left: 4px solid {severity_colors[sev]};">
        <div class="finding-header">
          <span class="badge" style="background:{severity_colors[sev]};">{sev.upper()}</span>
          <code>{f.file}:{f.line}</code>
          <span class="conf">confidence: {f.confidence.value}</span>
        </div>
        <h3>{f.type} <small>({f.id})</small></h3>
        <p>{f.ai_description}</p>
        {exploit_html}
        <p><strong>Remediation:</strong> {f.remediation}</p>
      </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SentinelAI Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1f2937; background: #f9fafb; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #6b7280; margin-bottom: 24px; }}
  .finding {{ background: white; border-radius: 6px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
  .finding-header {{ display: flex; align-items: center; gap: 10px; font-size: 13px; color: #6b7280; margin-bottom: 8px; }}
  .badge {{ color: white; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
  .finding h3 {{ margin: 4px 0; }}
  .conf {{ margin-left: auto; }}
  code {{ background: #f3f4f6; padding: 1px 6px; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>SentinelAI Vulnerability Report</h1>
  <p class="meta">Repository: <code>{repo_path}</code> &middot; Generated {generated} &middot; {len(findings)} findings</p>
  {"".join(rows)}
</body>
</html>"""
