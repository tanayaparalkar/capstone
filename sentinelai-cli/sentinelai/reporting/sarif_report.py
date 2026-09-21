"""
SARIF 2.1.0 report generator.

Builds a SARIF (Static Analysis Results Interchange Format) 2.1.0
document from an existing ScanResult and ScanStatistics, so SentinelAI
findings can be consumed by security tooling and CI/CD systems that
already understand SARIF (e.g. GitHub code scanning). Like the other
report generators, this does not recalculate statistics
(calculate_statistics() remains the single source of truth - the
already-computed ScanStatistics is only attached as-is under run-level
properties) or redefine finding fields - ScannerFinding/AIEnrichedFinding
remain the only source of truth.

Severity mapping (SentinelAI Severity -> SARIF result.level), chosen to
fit SARIF's four defined levels and documented here as the single source
of truth for the mapping:
    critical -> error
    high     -> error
    medium   -> warning
    low      -> note

Correlation: results are built from scanner_findings; a matching
AIEnrichedFinding (joined by finding_id, never by list position/length)
contributes an "ai" properties block to that one result. A finding with
no match gets no "ai" key at all - never a fabricated placeholder. AI
confidence never influences `level`, which is derived solely from the
scanner's own severity.

Rules: one SARIF rule per distinct rule_id actually present in this
scan, built only from scanner-level data (id, category, message, cwe) -
never from AI content, since one rule_id can be shared by findings with
different (or no) AI enrichment, and rule metadata must stay finding-
independent. Metadata for a given rule_id is taken from its first
occurrence in finding_id order (deterministic). rule_id is required by
the ScannerFinding contract, but as a defensive fallback for an empty
string, a deterministic id of the form "sentinelai/<finding_id>" is used
instead of leaving ruleId blank or inventing a random one.

Fingerprints: each result gets a `partialFingerprints` entry keyed on
finding_id, since finding_id is already a stable, deterministic per-
finding identifier - this lets SARIF consumers (e.g. GitHub code
scanning) track the same finding across repeated scans without
SentinelAI inventing any new identifier.

Validation note: this generator's output is checked two ways. Targeted
structural tests in tests/test_sarif_report.py assert the fields this
project depends on (schema/version, tool.driver, rules, results,
locations, levels), and tests/test_sarif_schema_conformance.py validates
emitted documents against the official OASIS sarif-schema-2.1.0.json.
That schema is vendored under tests/data/ and the validator is a dev-only
dependency, so neither adds weight to the installed package. Conformance
to the schema is not the same as acceptance by a particular code-scanning
platform, which remains untested.
"""
import json
from typing import Optional
from typing import Optional

from .. import __version__
from ..contracts import AIEnrichedFinding, ScanResult, ScannerFinding, Severity
from ..core import build_ai_lookup, build_correlation_lookup, normalize_file_uri
from ..statistics import ScanStatistics
from .models import PatchApplicationReport

SARIF_SCHEMA_URI = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"
SARIF_VERSION = "2.1.0"

_SEVERITY_TO_LEVEL = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
}


def build_sarif_report(result: ScanResult, statistics: ScanStatistics, patch_application: Optional[PatchApplicationReport] = None) -> dict:
    scanner_findings = sorted(result.scanner_findings, key=lambda f: f.finding_id)
    ai_by_id = build_ai_lookup(result)
    corr_by_id = build_correlation_lookup(result)

    rules, rule_index_by_id = _build_rules(scanner_findings)
    # One SARIF result per RAW finding: consumers must still see every location.
    results = [
        _build_result(f, ai_by_id.get(f.finding_id), rule_index_by_id, corr_by_id.get(f.finding_id))
        for f in scanner_findings
    ]

    # Bound rather than returned directly so the optional patch block can be
    # attached without duplicating the whole document for the two cases.
    report = {
        "$schema": SARIF_SCHEMA_URI,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "SentinelAI",
                        "fullName": "SentinelAI - AI-Powered Vulnerability Detection for DevSecOps",
                        "version": __version__,
                        "rules": rules,
                    }
                },
                "results": results,
                "properties": {
                    "sentinelai": {
                        "scannerTier": result.metadata.scanner_tier.value,
                        "statistics": statistics.model_dump(mode="json"),
                    }
                },
            }
        ],
    }

    # Under the RUN's sentinelai property bag - the same conservative placement
    # the AI fields already use. A consumer that knows only base SARIF sees an
    # ordinary static-analysis run; one that knows the namespace gets the patch
    # stage. Nothing is added to `results`, because a patch outcome is not a
    # finding and must not be mistaken for one.
    if patch_application is not None:
        report["runs"][0]["properties"]["sentinelai"]["patchApplication"] = (
            patch_application.model_dump(mode="json")
        )

    return report


def to_sarif(result: ScanResult, statistics: ScanStatistics, *, indent: int = 2,
             patch_application: Optional[PatchApplicationReport] = None) -> str:
    return json.dumps(build_sarif_report(result, statistics, patch_application), indent=indent)


def _rule_id(f: ScannerFinding) -> str:
    return f.rule_id if f.rule_id else f"sentinelai/{f.finding_id}"


def _build_rules(scanner_findings: list[ScannerFinding]) -> tuple:
    rules: list[dict] = []
    rule_index_by_id: dict = {}

    for f in scanner_findings:
        rule_id = _rule_id(f)
        if rule_id in rule_index_by_id:
            continue
        rule_index_by_id[rule_id] = len(rules)

        rule: dict = {
            "id": rule_id,
            "name": f.category,
            "shortDescription": {"text": f.message},
        }
        if f.cwe:
            rule["properties"] = {"cwe": f.cwe, "tags": [f.cwe]}
        rules.append(rule)

    return rules, rule_index_by_id


def _build_result(f: ScannerFinding, ai: Optional[AIEnrichedFinding], rule_index_by_id: dict, corr=None) -> dict:
    rule_id = _rule_id(f)
    result: dict = {
        "ruleId": rule_id,
        "ruleIndex": rule_index_by_id[rule_id],
        "level": _SEVERITY_TO_LEVEL[f.severity],
        "message": {"text": f.message},
        "partialFingerprints": {"sentinelaiFindingId/v1": f.finding_id},
    }

    location = _build_location(f)
    if location:
        result["locations"] = [location]

    properties: dict = {
        "findingId": f.finding_id,
        "scanner": f.scanner,
        "category": f.category,
        "severity": f.severity.value,
    }
    if f.cwe:
        properties["cwe"] = f.cwe
    if f.raw_evidence:
        properties["rawEvidence"] = f.raw_evidence
    if corr is not None:
        # Raw scanner ids are preserved above; these add the group view alongside.
        properties["correlationId"] = corr.correlation_id
        properties["canonicalFindingId"] = corr.canonical_finding_id
        properties["sourceFindingIds"] = corr.source_finding_ids
        properties["correlatedScanners"] = corr.scanners
        properties["correlationReason"] = corr.correlation_reason
    if ai is not None:
        properties["ai"] = _ai_properties(ai)

    result["properties"] = {"sentinelai": properties}

    return result


def _build_location(f: ScannerFinding) -> Optional[dict]:
    # SARIF line numbers are 1-based, matching ScannerFinding's own
    # (ge=1-validated) line_start/line_end directly - no conversion needed.
    # A dependency finding with no file is represented with no locations
    # entry at all, rather than an invented line.
    if not f.file:
        return None
    physical_location: dict = {"artifactLocation": {"uri": normalize_file_uri(f.file)}}
    if f.line_start:
        region = {"startLine": f.line_start}
        if f.line_end and f.line_end != f.line_start:
            region["endLine"] = f.line_end
        physical_location["region"] = region
    return {"physicalLocation": physical_location}


def _ai_properties(ai: AIEnrichedFinding) -> dict:
    props: dict = {
        "explanation": ai.explanation,
        "remediation": ai.remediation,
        "confidenceScore": ai.confidence_score,
        "confidenceLabel": ai.confidence_label.value,
        "verificationStatus": ai.verification_status.value,
    }
    if ai.exploit_path:
        props["exploitPath"] = ai.exploit_path
    if ai.impact:
        props["impact"] = ai.impact
    if ai.patch_suggestion:
        props["patchSuggestion"] = ai.patch_suggestion
    if ai.related_findings:
        props["relatedFindings"] = ai.related_findings
    if ai.references:
        props["references"] = ai.references
    # Critic review signals. Kept as distinct keys from verificationStatus above,
    # never merged with it: one is a deterministic category check, the other an
    # LLM grounding review. Absent keys mean no critic ran, matching how every
    # other optional AI field is emitted here.
    if ai.grounding_verdict is not None:
        props["groundingVerdict"] = ai.grounding_verdict.value
    if ai.supported_claims:
        props["supportedClaims"] = ai.supported_claims
    if ai.unsupported_claims:
        props["unsupportedClaims"] = ai.unsupported_claims
    return props
