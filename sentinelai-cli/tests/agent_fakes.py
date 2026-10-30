"""
Shared fakes for the multi-agent pipeline tests.

Not named test_*.py so pytest does not collect it as a test module.

make_agent_generate() returns a schema-aware stand-in for an LLMFn: it
dispatches on the `response_schema` the pipeline passes, so one fake can
serve all three agent stages with a schema-valid answer each. It records
every call, which is what lets the tests assert the two properties that
define this pipeline - that the stages run in dependency order, and that
exactly three generation calls happen per successful finding.

Dispatching on the schema rather than on call index is deliberate: an
index-based fake would still "pass" if the pipeline called the stages in
the wrong order, which is precisely the bug these tests exist to catch.
"""
import json
from typing import ClassVar, Optional

from sentinelai.ai.agents.schemas import (
    CriticAssessment,
    EvidenceAssessment,
    ExploitRemediationAssessment,
)

VALID_EVIDENCE = {
    "title": "sql-injection in db.py",
    "observed_evidence": ["A query string is built by concatenating a parameter."],
    "interpretation": "Concatenating input into a sql injection query allows statement manipulation.",
    "limitations": ["The evidence does not show whether the parameter is attacker-controlled."],
    "groundedness": "partially_supported",
}

VALID_EXPLOIT_REMEDIATION = {
    "exploit": {
        "exploit_path": "A crafted parameter value alters the query.",
        "impact": "Unauthorized data access.",
        "required_assumptions": ["The parameter is reachable from untrusted input."],
    },
    "remediation": {
        "remediation": "Use a parameterized sql injection-safe query API.",
        "patch_suggestion": "cursor.execute(sql, (value,))",
        "validation_steps": ["Re-run the scanner and confirm the finding is gone."],
    },
}

# The unsupported claim is deliberately a full sentence: CriticAssessment requires
# a non-supported verdict to name what is unsupported, in prose rather than as a
# bare identifier or location.
VALID_CRITIC = {
    "supported_claims": ["The query is built by string concatenation."],
    "unsupported_claims": [
        "The analysis assumes the parameter is attacker-controlled, which the evidence does not show."
    ],
    "verdict": "partially_supported",
}

from sentinelai.ai.agents import critic, evidence_analyst, exploit_remediation_analyst
from sentinelai.ai.llm import LLMProvider, LLMResponse

_BY_SCHEMA = {
    EvidenceAssessment: VALID_EVIDENCE,
    ExploitRemediationAssessment: VALID_EXPLOIT_REMEDIATION,
    CriticAssessment: VALID_CRITIC,
}


class RecordingGenerate:
    """Schema-aware fake LLMFn that records its calls.

    `calls` holds one (prompt, response_schema) tuple per invocation, in order.
    `schemas` is the convenience projection the ordering assertions use.
    """

    def __init__(self, fail_on_finding_ids=(), fail_on_schemas=(), overrides=None):
        self.calls = []
        self._fail_on_finding_ids = tuple(fail_on_finding_ids)
        self._fail_on_schemas = tuple(fail_on_schemas)
        self._overrides = overrides or {}

    @property
    def schemas(self):
        return [schema for _, schema in self.calls]

    def schemas_for(self, finding_id: str):
        return [schema for prompt, schema in self.calls if f"Finding: {finding_id}\n" in prompt]

    def __call__(self, prompt: str, response_schema=None) -> str:
        self.calls.append((prompt, response_schema))

        for finding_id in self._fail_on_finding_ids:
            if f"Finding: {finding_id}\n" in prompt:
                raise ConnectionRefusedError(f"simulated failure for {finding_id}")

        if response_schema in self._fail_on_schemas:
            raise ConnectionRefusedError(f"simulated failure for {response_schema.__name__}")

        if response_schema is None:
            raise AssertionError("multi-agent stages must always pass a response_schema")

        payload = self._overrides.get(response_schema, _BY_SCHEMA[response_schema])
        return json.dumps(payload)


def make_agent_generate(**kwargs) -> RecordingGenerate:
    return RecordingGenerate(**kwargs)


# Each agent's prompt opens with its own role text, which identifies the stage - and so
# the schema - without the response_schema argument LLMProvider.complete() does not take.
_SCHEMA_BY_ROLE = (
    (evidence_analyst._ROLE, EvidenceAssessment),
    (exploit_remediation_analyst._ROLE, ExploitRemediationAssessment),
    (critic._ROLE, CriticAssessment),
)


class GenerateBackedProvider(LLMProvider):
    """An LLMProvider over a RecordingGenerate, for driving the composed CLI path (main -> guard -> provider).

    Named "ollama" and local so it satisfies the egress guard under the default configuration.
    """

    name: ClassVar[str] = "ollama"
    is_local: ClassVar[bool] = True
    retains_data: ClassVar[Optional[bool]] = False

    def __init__(self, generate):
        self.generate = generate

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> LLMResponse:
        schema = next((s for role, s in _SCHEMA_BY_ROLE if prompt.startswith(role)), None)
        return LLMResponse(text=self.generate(prompt, response_schema=schema), model="fake-model")
