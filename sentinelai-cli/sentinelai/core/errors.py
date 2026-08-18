"""
Small error hierarchy.

This exists to give CLI code specific exceptions to catch, not to
multiply exit codes - several distinct exceptions can (and do) map to
the same ExitCode. A malformed saved report and a bad CLI flag are
different exceptions but the same *kind* of outcome (INVALID_INPUT):
both mean "the request as given can't be fulfilled," fixable by the
user without any code change.
"""


class SentinelAIError(Exception):
    """Base class for all SentinelAI errors the CLI knows how to handle cleanly."""


class InvalidInputError(SentinelAIError):
    """A bad CLI argument, an unusable input/output path, or a malformed/unsupported saved report."""


class ProviderError(SentinelAIError):
    """The FindingsProvider failed to produce a scan result."""


class AIEnrichmentError(SentinelAIError):
    """
    The optional AI enrichment layer failed once it was actually running.

    Covers the ways a configured AI layer can fail at runtime: an
    unreachable Ollama server, and - the common misconfiguration - an
    embedding model that cannot produce embeddings (see
    ai/embeddings_ollama.py). Raised so the CLI can report one clean,
    actionable line instead of letting urllib.error.HTTPError surface as
    a traceback.

    A sibling of ProviderError rather than a subclass: both mean "a thing
    SentinelAI depends on failed, not SentinelAI itself" and both map to
    ExitCode.PROVIDER_ERROR, but the AI layer is not a FindingsProvider,
    and code catching ProviderError around provider.get_scan_result()
    must not accidentally catch this.

    Note this is a *runtime* failure only. AI enrichment that is simply
    not configured is not an error at all and never reaches this class -
    main.py skips the whole stage and the scan succeeds scanner-only.
    """
