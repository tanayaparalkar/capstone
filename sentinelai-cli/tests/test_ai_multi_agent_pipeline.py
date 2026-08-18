"""
Multi-agent pipeline: stage ordering, call budget, structured-output
requests, and deterministic synthesis.

The properties pinned here are the ones that make the "multi-agent"
description true rather than decorative:

- three generation calls per successful finding, no more and no fewer;
- the stages run in dependency order, each receiving the previous
  stage's validated output;
- synthesis is ordinary Python that adds no calls and invents no data;
- scanner-owned fields survive untouched, and the critic's verdict never
  becomes the deterministic verification status or vice versa.

Ordering is asserted via the response_schema each call carries rather
than by call index, so a pipeline that ran the stages out of order would
fail rather than coincidentally pass.
"""
import json
from unittest.mock import patch

import pytest
from agent_fakes import (
    VALID_CRITIC,
    VALID_EVIDENCE,
    VALID_EXPLOIT_REMEDIATION,
    make_agent_generate,
)

from sentinelai.ai.agents.schemas import (
    CriticAssessment,
    EvidenceAssessment,
    ExploitAssessment,
    ExploitRemediationAssessment,
    RemediationPlan,
)
from sentinelai.ai.agents.synthesizer import build_verifier_input, synthesize
from sentinelai.ai.config import AISettings
from sentinelai.ai.confidence_scorer import score_confidence
from sentinelai.ai.llm_ollama import OllamaProvider
from sentinelai.ai.pipeline import enrich_findings
from sentinelai.ai.repository_context import RepositoryContext
from sentinelai.ai.retrieval import RetrievedChunk, Retriever
from sentinelai.ai.verifier import verify_finding
from sentinelai.contracts import (
    ConfidenceLabel,
    GroundingVerdict,
    ScannerFinding,
    Severity,
    VerificationStatus,
)

HOST = "http://localhost:11434"


def _finding(finding_id: str = "SENT-001", **overrides) -> ScannerFinding:
    base = dict(
        finding_id=finding_id,
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="python.sql.concat",
        message="test finding",
        raw_evidence="query = 'SELECT ' + x",
    )
    base.update(overrides)
    return ScannerFinding(**base)


class _FixedRetriever(Retriever):
    def __init__(self, chunks=None):
        self._chunks = chunks if chunks is not None else [
            RetrievedChunk(
                kb_entry_id="KB-1", text="sql injection", score=1.0, references=["https://cwe.mitre.org/89"]
            )
        ]

    def retrieve(self, query, top_k):
        return list(self._chunks)


def _settings(monkeypatch, **overrides):
    monkeypatch.setattr("sentinelai.ai.pipeline.get_settings", lambda: AISettings(**overrides))


def _run(monkeypatch, findings, generate=None, retriever=None, **settings_overrides):
    _settings(monkeypatch, **settings_overrides)
    generate = generate or make_agent_generate()
    results = enrich_findings(
        findings, retriever or _FixedRetriever(), generate, score_confidence, verify_finding
    )
    return results, generate


# --- call budget -------------------------------------------------------------------------------------------------------


def test_exactly_three_llm_calls_per_successful_finding(monkeypatch):
    results, generate = _run(monkeypatch, [_finding()])

    assert len(results) == 1
    assert len(generate.calls) == 3


def test_call_budget_scales_linearly_with_findings(monkeypatch):
    findings = [_finding(f"SENT-00{i}") for i in range(1, 4)]

    results, generate = _run(monkeypatch, findings)

    assert len(results) == 3
    assert len(generate.calls) == 9  # 3 findings x 3 calls, no batching, no extra calls


def test_synthesis_adds_no_llm_call(monkeypatch):
    # The third call is the critic; nothing after it may reach the model.
    _, generate = _run(monkeypatch, [_finding()])

    assert generate.schemas[-1] is CriticAssessment


# --- stage ordering ----------------------------------------------------------------------------------------------------


def test_stages_run_in_dependency_order(monkeypatch):
    _, generate = _run(monkeypatch, [_finding()])

    assert generate.schemas == [EvidenceAssessment, ExploitRemediationAssessment, CriticAssessment]


def test_each_finding_gets_its_own_ordered_stage_sequence(monkeypatch):
    findings = [_finding("SENT-001"), _finding("SENT-002")]

    _, generate = _run(monkeypatch, findings)

    for finding_id in ("SENT-001", "SENT-002"):
        assert generate.schemas_for(finding_id) == [
            EvidenceAssessment,
            ExploitRemediationAssessment,
            CriticAssessment,
        ]


def test_later_stages_receive_earlier_stage_output(monkeypatch):
    _, generate = _run(monkeypatch, [_finding()])
    _, exploit_prompt = generate.calls[1][0], generate.calls[1][0]
    critic_prompt = generate.calls[2][0]

    # Stage 2 is shown the evidence analyst's interpretation...
    assert VALID_EVIDENCE["interpretation"] in exploit_prompt
    assert "Prior evidence assessment" in exploit_prompt
    # ...and stage 3 is shown the claims from stages 1 and 2.
    assert "Claims under review" in critic_prompt
    assert VALID_EXPLOIT_REMEDIATION["remediation"]["remediation"] in critic_prompt


def test_every_stage_requests_a_constrained_schema(monkeypatch):
    _, generate = _run(monkeypatch, [_finding()])

    assert all(schema is not None for schema in generate.schemas)


# --- structured output: the `format` field ------------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _sent_payload(urlopen_mock) -> dict:
    return json.loads(urlopen_mock.call_args[0][0].data.decode("utf-8"))


def test_request_includes_format_only_when_a_schema_is_supplied():
    provider = OllamaProvider(host=HOST, model="llama3.1:8b")
    body = json.dumps({"response": json.dumps(VALID_CRITIC)})

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)) as urlopen:
        provider.generate("prompt")
    assert "format" not in _sent_payload(urlopen)

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)) as urlopen:
        provider.generate("prompt", response_schema=CriticAssessment)
    payload = _sent_payload(urlopen)
    assert payload["format"] == CriticAssessment.model_json_schema()


def test_format_carries_the_nested_schema_for_the_combined_call():
    provider = OllamaProvider(host=HOST, model="llama3.1:8b")
    body = json.dumps({"response": json.dumps(VALID_EXPLOIT_REMEDIATION)})

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)) as urlopen:
        provider.generate("prompt", response_schema=ExploitRemediationAssessment)

    assert "$defs" in _sent_payload(urlopen)["format"]


def test_unconstrained_request_is_otherwise_unchanged():
    provider = OllamaProvider(host=HOST, model="llama3.1:8b")
    body = json.dumps({"response": "plain text"})

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)) as urlopen:
        provider.generate("prompt")

    payload = _sent_payload(urlopen)
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.0}
    assert set(payload) == {"model", "prompt", "stream", "options"}


# --- deterministic synthesis ---------------------------------------------------------------------------------------------


def _assessments():
    return (
        EvidenceAssessment.model_validate(VALID_EVIDENCE),
        ExploitRemediationAssessment.model_validate(VALID_EXPLOIT_REMEDIATION),
        CriticAssessment.model_validate(VALID_CRITIC),
    )


def _synthesize(finding=None, **overrides):
    evidence, assessment, critic = _assessments()
    kwargs = dict(
        finding=finding or _finding(),
        evidence=evidence,
        assessment=assessment,
        critic=critic,
        retrieved_chunks=[
            RetrievedChunk(kb_entry_id="KB-1", text="t", score=1.0, references=["https://cwe.mitre.org/89"])
        ],
        confidence_score=0.87,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
        repository_context=RepositoryContext(summary="Repository 'demo'."),
    )
    kwargs.update(overrides)
    return synthesize(**kwargs)


def test_synthesis_preserves_scanner_owned_fields():
    finding = _finding(severity=Severity.CRITICAL, scanner="bandit", raw_evidence="dangerous()")

    result = _synthesize(finding=finding)

    assert result.finding_id == finding.finding_id
    assert result.severity is Severity.CRITICAL       # scanner's severity, not the agents'
    assert result.scanner_sources == ["bandit"]
    assert result.evidence == "dangerous()"


def test_synthesis_passes_confidence_and_verification_through_untouched():
    result = _synthesize(confidence_score=0.42, confidence_label=ConfidenceLabel.MEDIUM)

    assert result.confidence_score == 0.42
    assert result.confidence_label is ConfidenceLabel.MEDIUM
    assert result.verification_status is VerificationStatus.VERIFIED


def test_synthesis_keeps_grounding_verdict_separate_from_verification_status():
    # The two signals must never be derived from one another - they are allowed
    # to disagree, and a disagreement is information.
    result = _synthesize(verification_status=VerificationStatus.REJECTED)

    assert result.verification_status is VerificationStatus.REJECTED
    assert result.grounding_verdict is GroundingVerdict.PARTIALLY_SUPPORTED


def test_synthesis_carries_critic_claims_onto_the_contract():
    result = _synthesize()

    assert result.supported_claims == VALID_CRITIC["supported_claims"]
    assert result.unsupported_claims == VALID_CRITIC["unsupported_claims"]


def test_synthesis_takes_references_from_the_knowledge_base_only():
    result = _synthesize()

    assert result.references == ["https://cwe.mitre.org/89"]


def test_synthesis_labels_assumptions_inside_the_exploit_path():
    result = _synthesize()

    assert "Assumptions required" in result.exploit_path
    assert VALID_EXPLOIT_REMEDIATION["exploit"]["required_assumptions"][0] in result.exploit_path


def test_synthesis_labels_limitations_inside_the_explanation():
    result = _synthesize()

    assert "Limitations of the available evidence" in result.explanation
    assert VALID_EVIDENCE["limitations"][0] in result.explanation


def test_synthesis_omits_exploit_path_when_none_was_established():
    assessment = ExploitRemediationAssessment(
        exploit=ExploitAssessment(required_assumptions=["unused without a path"]),
        remediation=RemediationPlan(remediation="fix"),
    )

    result = _synthesize(assessment=assessment)

    assert result.exploit_path is None
    assert result.impact is None


def test_synthesis_allows_a_null_patch_suggestion():
    assessment = ExploitRemediationAssessment(
        exploit=ExploitAssessment(),
        remediation=RemediationPlan(remediation="fix", patch_suggestion=None),
    )

    assert _synthesize(assessment=assessment).patch_suggestion is None


def test_synthesis_is_deterministic():
    assert _synthesize() == _synthesize()


def test_synthesis_requires_a_scanner_finding():
    # There is no code path by which the AI layer can emit an enrichment for a
    # finding no scanner reported: the ScannerFinding is a required argument.
    with pytest.raises(TypeError):
        synthesize(
            evidence=_assessments()[0],
            assessment=_assessments()[1],
            critic=_assessments()[2],
            retrieved_chunks=[],
            confidence_score=0.5,
            confidence_label=ConfidenceLabel.MEDIUM,
            verification_status=VerificationStatus.UNVERIFIED,
            repository_context=RepositoryContext(),
        )


def test_verifier_input_matches_the_synthesized_narrative():
    # The deterministic verifier must judge the same text a reader sees.
    evidence, assessment, _ = _assessments()

    adapted = build_verifier_input(evidence, assessment)
    result = _synthesize()

    assert adapted.explanation == result.explanation
    assert adapted.remediation == result.remediation
    assert adapted.exploit_path == result.exploit_path


# --- failure isolation across the three stages ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failing_stage",
    [EvidenceAssessment, ExploitRemediationAssessment, CriticAssessment],
    ids=["evidence", "exploit_remediation", "critic"],
)
def test_a_failure_at_any_stage_skips_only_that_finding(monkeypatch, failing_stage):
    # Whichever stage breaks, the finding is omitted rather than half-enriched.
    _settings(monkeypatch)
    generate = make_agent_generate(fail_on_schemas=(failing_stage,))

    from sentinelai.core.errors import AIEnrichmentError

    with pytest.raises(AIEnrichmentError):
        enrich_findings(
            [_finding()], _FixedRetriever(), generate, score_confidence, verify_finding
        )


def test_a_malformed_stage_response_is_not_silently_accepted(monkeypatch):
    _settings(monkeypatch)
    generate = make_agent_generate(overrides={CriticAssessment: {"verdict": "definitely_fine"}})

    from sentinelai.core.errors import AIEnrichmentError

    with pytest.raises(AIEnrichmentError):
        enrich_findings([_finding()], _FixedRetriever(), generate, score_confidence, verify_finding)


def test_verification_can_still_be_disabled(monkeypatch):
    results, _ = _run(monkeypatch, [_finding()], enable_verification=False)

    assert results[0].verification_status is VerificationStatus.UNVERIFIED
    assert results[0].grounding_verdict is not None  # critic still ran and is reported


# --- critic verdict/claim invariant is enforced end-to-end -------------------------------------------------------------
#
# The live defect this guards: the critic returned a partially_supported or
# insufficient_evidence verdict with an empty unsupported_claims list on 17/17
# findings. That validated under the old schema and reached synthesis, producing
# a verdict badge in every report with no explanation beneath it. It must now
# fail the finding instead.


@pytest.mark.parametrize(
    "verdict", ["partially_supported", "insufficient_evidence"], ids=["partial", "insufficient"]
)
def test_unexplained_non_supported_verdict_fails_the_finding(monkeypatch, verdict):
    from sentinelai.core.errors import AIEnrichmentError

    _settings(monkeypatch)
    generate = make_agent_generate(
        overrides={CriticAssessment: {"verdict": verdict, "supported_claims": [], "unsupported_claims": []}}
    )

    with pytest.raises(AIEnrichmentError):
        enrich_findings([_finding()], _FixedRetriever(), generate, score_confidence, verify_finding)


@pytest.mark.parametrize(
    "verdict", ["partially_supported", "insufficient_evidence"], ids=["partial", "insufficient"]
)
def test_explained_non_supported_verdict_reaches_synthesis(monkeypatch, verdict):
    _settings(monkeypatch)
    claim = "The analysis assumes attacker control the evidence does not establish."
    generate = make_agent_generate(
        overrides={
            CriticAssessment: {
                "verdict": verdict,
                "supported_claims": [],
                "unsupported_claims": [claim],
            }
        }
    )

    results = enrich_findings(
        [_finding()], _FixedRetriever(), generate, score_confidence, verify_finding
    )

    assert results[0].grounding_verdict.value == verdict
    assert results[0].unsupported_claims == [claim]


def test_supported_verdict_with_unsupported_claims_fails_the_finding(monkeypatch):
    # The invariant read from the other direction.
    from sentinelai.core.errors import AIEnrichmentError

    _settings(monkeypatch)
    generate = make_agent_generate(
        overrides={
            CriticAssessment: {
                "verdict": "supported",
                "supported_claims": [],
                "unsupported_claims": ["Something the evidence does not actually support here."],
            }
        }
    )

    with pytest.raises(AIEnrichmentError):
        enrich_findings([_finding()], _FixedRetriever(), generate, score_confidence, verify_finding)


def test_a_location_as_an_unsupported_claim_fails_the_finding(monkeypatch):
    from sentinelai.core.errors import AIEnrichmentError

    _settings(monkeypatch)
    generate = make_agent_generate(
        overrides={
            CriticAssessment: {
                "verdict": "partially_supported",
                "supported_claims": [],
                "unsupported_claims": ["app/db.py:7"],
            }
        }
    )

    with pytest.raises(AIEnrichmentError):
        enrich_findings([_finding()], _FixedRetriever(), generate, score_confidence, verify_finding)
