"""AIConfig defaults and ScanMetadata backward compatibility with pre-AI-provider reports."""
import pytest
from pydantic import ValidationError

from sentinelai.contracts import AIConfig, ScanMetadata, ScanResult


class TestAIConfigDefaults:
    def test_egress_defaults_to_deny(self):
        assert AIConfig(provider="groq", model="any").egress == "deny"

    @pytest.mark.parametrize("provider", ["ollama", "groq", "gemini", "mock"])
    def test_egress_defaults_to_deny_for_every_provider(self, provider):
        assert AIConfig(provider=provider, model="any").egress == "deny"

    def test_remaining_defaults(self):
        config = AIConfig(provider="mock", model="any")
        assert config.redaction_enabled is True
        assert config.max_context_lines == 10
        assert config.max_context_chars == 2000

    def test_unknown_egress_value_is_rejected(self):
        with pytest.raises(ValidationError):
            AIConfig(provider="mock", model="any", egress="maybe")

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValidationError):
            AIConfig(provider="openai", model="any")


# A ScanResult exactly as written before the AI provider fields existed:
# metadata carries none of ai_provider, ai_model, egress_policy, redaction.
LEGACY_REPORT = {
    "repository": {"name": "demo-app", "path": "/tmp/demo-app"},
    "metadata": {
        "timestamp": "2026-08-08T12:00:00Z",
        "mode": "standard",
        "duration_seconds": 12.4,
        "scanner_tier": "core",
    },
    "scanner_findings": [],
    "ai_findings": [],
    "correlated_findings": [],
}

NEW_FIELDS = ("ai_provider", "ai_model", "egress_policy", "retention_status", "redaction")


class TestScanMetadataBackwardCompatibility:
    def test_legacy_metadata_loads_with_new_fields_none(self):
        metadata = ScanMetadata.model_validate(LEGACY_REPORT["metadata"])
        for field in NEW_FIELDS:
            assert getattr(metadata, field) is None

    def test_legacy_report_loads_and_round_trips(self):
        result = ScanResult.model_validate(LEGACY_REPORT)
        assert ScanResult.model_validate_json(result.model_dump_json()) == result

    def test_minimal_construction_still_works(self):
        metadata = ScanMetadata(timestamp="2026-08-08T12:00:00Z", mode="quick")
        for field in NEW_FIELDS:
            assert getattr(metadata, field) is None
