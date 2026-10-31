"""
Privacy and egress enforcement on the embedding path (P1).

What this protects. ai/retrieval/context_retrieval.py builds its retrieval
query from a finding's category, message and raw_evidence - raw_evidence being
a verbatim excerpt of the scanned repository - and ai/embeddings_ollama.py
posts it to /api/embed. Before P1 that text was serialized exactly as built:
no redaction, no bounds, and no egress assertion. Measured on the benchmark
fixture, 2 of 23 queries carried content the project's own egress gate
classifies as a secret.

Why the assertions look at the request body. A test that only checked the
return value of embed() would pass whether or not redaction happened. Every
test here captures the bytes handed to urllib and asserts on those, because
the body is the thing that would leave the process.

No live endpoint, no real credential: urlopen is mocked throughout and every
secret is synthetic.
"""
import json
import urllib.request
from unittest.mock import patch

import pytest

from sentinelai.ai.embeddings_ollama import (
    _MAX_BATCH_TEXTS,
    _MAX_TOTAL_CHARS,
    make_ollama_embed_fn,
)
from sentinelai.ai.egress import EgressViolation
from sentinelai.ai.local_endpoint import RemoteEndpointError
from sentinelai.contracts import AIConfig, ScannerFinding, Severity
from sentinelai.core.errors import AIEnrichmentError

HOST = "http://localhost:11434"
MODEL = "nomic-embed-text"

# Synthetic throughout - none of these is a real credential.
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
PASSWORD = "hunter2" + "SuperSecretValue"


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capturing_urlopen(sink: dict):
    """Records the outbound body and answers with one vector per input text."""

    def fake(request, timeout=None):
        sink["body"] = request.data.decode("utf-8")
        sink["parsed"] = json.loads(sink["body"])
        count = len(sink["parsed"]["input"])
        return _Response(json.dumps({"embeddings": [[0.1, 0.2]] * count}).encode("utf-8"))

    return fake


def _embed(texts, sink=None):
    sink = {} if sink is None else sink
    with patch("urllib.request.urlopen", side_effect=_capturing_urlopen(sink)) as urlopen:
        vectors = make_ollama_embed_fn(host=HOST, model=MODEL)(texts)
    return vectors, sink, urlopen


# ------------------------------------------------- 1 & 2: redaction first

def test_a_secret_is_redacted_before_the_body_is_built():
    vectors, sink, _ = _embed([f'aws_access_key_id = "{AWS_KEY}"'])
    assert len(vectors) == 1
    assert AWS_KEY not in sink["body"], "the secret reached the request body"
    assert "REDACTED" in sink["body"], "nothing was masked"


@pytest.mark.parametrize(
    "text",
    [
        f'aws_access_key_id = "{AWS_KEY}"',
        f'PASSWORD = "{PASSWORD}"',
        f'conn = "postgresql://app:{PASSWORD}@db.internal:5432/app"',
        f'headers = {{"Authorization": "Bearer {PASSWORD}"}}',
    ],
)
def test_no_synthetic_secret_appears_in_the_serialized_payload(text):
    _, sink, _ = _embed([text])
    assert AWS_KEY not in sink["body"]
    assert PASSWORD not in sink["body"]


def test_the_redacted_text_is_what_gets_sent_not_the_original():
    original = f'key = "{AWS_KEY}"'
    _, sink, _ = _embed([original])
    assert sink["parsed"]["input"] != [original]
    assert len(sink["parsed"]["input"]) == 1


def test_the_scan_session_secret_index_is_reused_when_one_exists():
    """Inside a scan the query is masked against the same index the prompts use."""
    from sentinelai.ai.prompt_builder import scan_session

    finding = ScannerFinding(
        finding_id="gitleaks-0",
        scanner="gitleaks",
        category="generic-api-key",
        severity=Severity.HIGH,
        rule_id="generic-api-key",
        message="hardcoded credential",
        file="config/settings.py",
        line_start=4,
        raw_evidence=f'API_KEY = "{AWS_KEY}"',
    )
    cfg = AIConfig(provider="ollama", model=MODEL, egress="deny")
    with scan_session([finding], cfg) as session:
        _, sink, _ = _embed([f'API_KEY = "{AWS_KEY}"'])
        assert AWS_KEY not in sink["body"]
        # The embedding text is accounted for in the scan's redaction report.
        assert session.redaction_report().secrets_masked >= 1


# ----------------------------------------------------------- 3: bounds

def test_an_over_long_text_is_truncated_deterministically():
    long_text = "A" * 10_000
    _, sink, _ = _embed([long_text])
    sent = sink["parsed"]["input"][0]
    limit = AIConfig.model_fields["max_context_chars"].default
    assert len(sent) == limit
    assert sent == long_text[:limit], "truncation must be a prefix, not a sample"


def test_truncation_is_stable_across_calls():
    long_text = "B" * 5_000
    _, first, _ = _embed([long_text])
    _, second, _ = _embed([long_text])
    assert first["parsed"]["input"] == second["parsed"]["input"]


def test_text_already_within_bounds_is_untouched():
    text = "def login(user): return user.password"
    _, sink, _ = _embed([text])
    assert sink["parsed"]["input"] == [text]


# ------------------------------------------- 4 & 5: batch and payload caps

def test_an_oversized_batch_is_rejected_before_any_request():
    texts = ["x"] * (_MAX_BATCH_TEXTS + 1)
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(AIEnrichmentError) as excinfo:
            make_ollama_embed_fn(host=HOST, model=MODEL)(texts)
    urlopen.assert_not_called()
    assert str(_MAX_BATCH_TEXTS) in str(excinfo.value)


def test_a_batch_at_the_limit_is_accepted():
    _, sink, _ = _embed(["x"] * _MAX_BATCH_TEXTS)
    assert len(sink["parsed"]["input"]) == _MAX_BATCH_TEXTS


def test_an_oversized_total_payload_is_rejected_before_any_request():
    # Each text is within the per-text limit; together they exceed the total.
    limit = AIConfig.model_fields["max_context_chars"].default
    count = (_MAX_TOTAL_CHARS // limit) + 2
    assert count <= _MAX_BATCH_TEXTS, "this case must exercise the total cap, not the batch cap"
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(AIEnrichmentError) as excinfo:
            make_ollama_embed_fn(host=HOST, model=MODEL)(["C" * limit] * count)
    urlopen.assert_not_called()
    assert str(_MAX_TOTAL_CHARS) in str(excinfo.value)


def test_the_total_bound_and_the_gate_agree_at_the_boundary():
    """The explicit check counts the separators the gate will also count.

    "\\n".join(bounded) adds len(bounded) - 1 characters. Before this was
    accounted for, a batch summing to exactly _MAX_TOTAL_CHARS passed the
    explicit check and was then refused by the gate as a size violation - an
    egress fault reported for what is really a bounds overrun. 128 texts of 256
    characters is 32,768 summed but 32,895 joined.
    """
    count, each = _MAX_BATCH_TEXTS, _MAX_TOTAL_CHARS // _MAX_BATCH_TEXTS
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(AIEnrichmentError) as excinfo:
            make_ollama_embed_fn(host=HOST, model=MODEL)(["D" * each] * count)
    urlopen.assert_not_called()
    assert not isinstance(excinfo.value, EgressViolation), "the gate reported a bounds overrun"
    # The reported figure is the joined length the gate would have measured.
    assert str(count * each + count - 1) in str(excinfo.value)


def test_a_batch_just_inside_the_joined_limit_is_accepted():
    """One character smaller must still go through, so the bound is not over-tight."""
    count = _MAX_BATCH_TEXTS
    each = (_MAX_TOTAL_CHARS - (count - 1)) // count
    _, sink, _ = _embed(["E" * each] * count)
    joined = len("\n".join(sink["parsed"]["input"]))
    assert joined <= _MAX_TOTAL_CHARS
    assert len(sink["parsed"]["input"]) == count


def test_bounds_errors_name_no_payload_content():
    with patch("urllib.request.urlopen"):
        with pytest.raises(AIEnrichmentError) as excinfo:
            make_ollama_embed_fn(host=HOST, model=MODEL)([f'k="{AWS_KEY}"'] * (_MAX_BATCH_TEXTS + 1))
    assert AWS_KEY not in str(excinfo.value)


# ------------------------------------- a zero context limit is refused

def _with_limit(limit: int):
    """A scan session whose max_context_chars is `limit`."""
    from sentinelai.ai.prompt_builder import scan_session

    return scan_session([], AIConfig(provider="ollama", model=MODEL, egress="deny", max_context_chars=limit))


def test_a_zero_context_limit_is_rejected_and_sends_nothing():
    """text[:0] would empty every input and still post an embedding request.

    AIConfig.max_context_chars is Field(2000, ge=0), so zero is contract-legal.
    Silently discarding the input and returning a vector for content that was
    never submitted is the failure this refuses.
    """
    with _with_limit(0):
        with patch("urllib.request.urlopen") as urlopen:
            with pytest.raises(AIEnrichmentError) as excinfo:
                make_ollama_embed_fn(host=HOST, model=MODEL)(["a real query"])
    urlopen.assert_not_called()
    assert "max_context_chars" in str(excinfo.value)


def test_the_zero_limit_error_leaks_no_input_or_secret():
    with _with_limit(0):
        with patch("urllib.request.urlopen"):
            with pytest.raises(AIEnrichmentError) as excinfo:
                make_ollama_embed_fn(host=HOST, model=MODEL)([f'key = "{AWS_KEY}"'])
    message = str(excinfo.value)
    assert AWS_KEY not in message
    assert "key =" not in message


def test_a_positive_limit_still_works():
    with _with_limit(64):
        _, sink, _ = _embed(["F" * 500])
    assert len(sink["parsed"]["input"][0]) == 64


# --------------------------------------------- 6 & 7: egress enforcement

def test_a_non_local_embedding_destination_is_refused_before_network_access(monkeypatch):
    """The gate is wired in, and it fails closed.

    The embedding leg is described to the gate as provider=ollama with
    egress=deny, which passes only because OllamaProvider.is_local is True.
    Flipping that models a destination that is not on this machine; the gate
    must then refuse before a socket is opened.
    """
    from sentinelai.ai.llm.ollama import OllamaProvider

    monkeypatch.setattr(OllamaProvider, "is_local", False)
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(EgressViolation) as excinfo:
            make_ollama_embed_fn(host=HOST, model=MODEL)(["anything"])
    urlopen.assert_not_called()
    assert excinfo.value.check == "policy"


def test_a_direct_call_cannot_bypass_redaction_or_the_gate():
    """make_ollama_embed_fn is public and callable without the pipeline.

    Enforcement lives inside embed() rather than in a wrapper precisely so this
    path is covered too.
    """
    _, sink, _ = _embed([f'secret = "{AWS_KEY}"'])
    assert AWS_KEY not in sink["body"]

    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(AIEnrichmentError):
            make_ollama_embed_fn(host=HOST, model=MODEL)(["y"] * (_MAX_BATCH_TEXTS + 1))
    urlopen.assert_not_called()


def test_an_empty_batch_still_short_circuits_without_a_request():
    with patch("urllib.request.urlopen") as urlopen:
        assert make_ollama_embed_fn(host=HOST, model=MODEL)([]) == []
    urlopen.assert_not_called()


# ------------------------------------ 8 & 12: real workloads still work

def test_the_bundled_knowledge_base_still_indexes():
    """10 entries, ~5,260 characters - the largest batch this project sends."""
    from sentinelai.ai.retrieval.in_memory_retriever import _entry_to_text
    from sentinelai.security_kb import load_knowledge_base

    entries = load_knowledge_base()
    texts = [_entry_to_text(e) for e in entries]
    assert len(entries) == 10
    vectors, sink, _ = _embed(texts)
    assert len(vectors) == len(entries)
    assert len(sink["parsed"]["input"]) == len(entries)


def test_retrieval_still_returns_ranked_chunks():
    """Normal retrieval semantics are unchanged by the new enforcement."""
    from sentinelai.ai.retrieval.in_memory_retriever import InMemoryRetriever
    from sentinelai.security_kb import load_knowledge_base

    sink = {}
    with patch("urllib.request.urlopen", side_effect=_capturing_urlopen(sink)):
        embed = make_ollama_embed_fn(host=HOST, model=MODEL)
        retriever = InMemoryRetriever(load_knowledge_base(), embed)
        chunks = retriever.retrieve("hardcoded credential in settings", top_k=3)
    assert len(chunks) == 3
    assert all(chunk.kb_entry_id for chunk in chunks)
    scores = [chunk.score for chunk in chunks]
    assert scores == sorted(scores, reverse=True), "chunks must stay ranked"


# ----------------- P2: positional GitLeaks secrets in embedding queries

# A synthetic high-entropy value with no keyword around it, so only the
# positional index can find it - the regex patterns have nothing to anchor on.
POSITIONAL_SECRET = "Zm9vYmFy" + "YmF6cXV4MTIzNDU2Nzg5"

# Shapes measured in the P2 audit. Only the assignment form is caught by the
# regex patterns; the rest rely entirely on the positional SecretIndex.
_LEAKING_SHAPES = {
    "function_arg": 'x = 1\nauthenticate("{s}")\ny = 2',
    "dict_literal": 'x = 1\ncfg = {{"tok": "{s}"}}\ny = 2',
    "list_element": 'x = 1\nKEYS = ["{s}"]\ny = 2',
    "bare_string": 'x = 1\n"{s}"\ny = 2',
    "f_string": 'x = 1\nurl = f"https://api.example.com/{{\'{s}\'}}"\ny = 2',
    "assignment": 'x = 1\ntoken = "{s}"\ny = 2',
}


def _gitleaks_finding(path="config/settings.py", line=2):
    """A GitLeaks finding as the scanner really emits it: evidence already --redact'ed."""
    return ScannerFinding(
        finding_id="gl-0",
        scanner="gitleaks",
        category="generic-api-key",
        severity=Severity.HIGH,
        rule_id="generic-api-key",
        message="secret detected",
        file=path,
        line_start=line,
        line_end=line,
        raw_evidence="REDACTED",
    )


def _bandit_finding(evidence, path="config/settings.py", first=1, last=3):
    """A Bandit finding whose evidence window spans the GitLeaks-flagged line."""
    return ScannerFinding(
        finding_id="bandit-0",
        scanner="bandit",
        category="hardcoded_password_string",
        severity=Severity.MEDIUM,
        rule_id="B105",
        message="possible hardcoded password",
        file=path,
        line_start=first,
        line_end=last,
        raw_evidence=evidence,
    )


def _query_for(findings, target):
    from sentinelai.ai.prompt_builder import scan_session
    from sentinelai.ai.repository_context import RepositoryContext
    from sentinelai.ai.retrieval.context_retrieval import _build_query

    cfg = AIConfig(provider="ollama", model=MODEL, egress="deny")
    with scan_session(findings, cfg) as session:
        return _build_query(target, RepositoryContext()), session


@pytest.mark.parametrize("shape", sorted(_LEAKING_SHAPES))
def test_a_positionally_flagged_secret_is_masked_in_the_query(shape):
    """The exposure: a Bandit finding whose evidence spans a GitLeaks-flagged line.

    That evidence is not --redact'ed by anyone - a different scanner produced it -
    so before P2 five of these six shapes carried the secret into the embedding
    request. Assertions use a boolean so a failure never prints the value.
    """
    evidence = _LEAKING_SHAPES[shape].format(s=POSITIONAL_SECRET)
    bandit = _bandit_finding(evidence)
    query, _ = _query_for([_gitleaks_finding(), bandit], bandit)
    assert POSITIONAL_SECRET not in query, f"{shape}: secret survived into the query"
    assert "REDACTED" in query


@pytest.mark.parametrize("shape", sorted(_LEAKING_SHAPES))
def test_the_serialized_request_carries_the_marker_not_the_secret(shape):
    """End to end: query construction through to the bytes handed to urllib."""
    evidence = _LEAKING_SHAPES[shape].format(s=POSITIONAL_SECRET)
    bandit = _bandit_finding(evidence)
    query, _ = _query_for([_gitleaks_finding(), bandit], bandit)
    _, sink, _ = _embed([query])
    assert POSITIONAL_SECRET not in sink["body"], f"{shape}: secret reached the request body"
    assert "REDACTED" in sink["body"]


def test_it_works_when_the_literal_store_is_empty():
    """GitLeaks --redact means literals stay empty; positional ranges do the work."""
    from sentinelai.ai.redaction import SecretIndex

    index = SecretIndex()
    index.add_from_findings([_gitleaks_finding()])
    assert "literals=0" in repr(index), "this test is meaningless if literals are populated"

    bandit = _bandit_finding(_LEAKING_SHAPES["bare_string"].format(s=POSITIONAL_SECRET))
    query, _ = _query_for([_gitleaks_finding(), bandit], bandit)
    assert POSITIONAL_SECRET not in query


def test_a_value_outside_the_flagged_range_is_not_over_masked():
    """Only the flagged line is positional; unrelated lines must survive intact."""
    marker_text = "KEEP_THIS_LINE_INTACT"
    evidence = f'{marker_text} = 1\n"{POSITIONAL_SECRET}"\nother = 2'
    bandit = _bandit_finding(evidence, first=1, last=3)
    query, _ = _query_for([_gitleaks_finding(path="config/settings.py", line=2), bandit], bandit)
    assert POSITIONAL_SECRET not in query
    assert marker_text in query, "an unflagged line was masked"
    assert "other = 2" in query


def test_a_clean_finding_produces_byte_identical_query_text():
    """A finding in a different file from the flagged one must be untouched."""
    evidence = "def add(a, b):\n    return a + b"
    bandit = _bandit_finding(evidence, path="app/math_utils.py", first=1, last=2)
    query, _ = _query_for([_gitleaks_finding(path="config/settings.py", line=2), bandit], bandit)
    assert evidence in query, "a clean finding's evidence was altered"
    assert "REDACTED" not in query


def test_a_flagged_line_is_masked_even_when_the_excerpt_looks_clean():
    """Documents real, intended behaviour rather than asserting it away.

    GitLeaks runs with --redact, so its evidence is the literal word REDACTED
    and offers no template to locate the secret on its line. redactor.py then
    masks the whole flagged line - fail closed. So a Bandit excerpt that spans
    that line has the line replaced even though the excerpt itself shows no
    secret: the index knows a credential lives there, and the excerpt may be
    stale. The prompt path has always behaved this way
    (prompt_builder._format_finding_section); P2 only brings the embedding query
    into line with it.
    """
    evidence = "def add(a, b):\n    return a + b\nresult = add(1, 2)"
    bandit = _bandit_finding(evidence, path="config/settings.py", first=1, last=3)
    query, _ = _query_for([_gitleaks_finding(path="config/settings.py", line=2), bandit], bandit)
    assert "REDACTED" in query, "the flagged line should be masked"
    assert "def add(a, b):" in query, "unflagged lines must survive"
    assert "result = add(1, 2)" in query


def test_redaction_is_idempotent_and_does_not_double_count():
    """_prepare_texts redacts again; that second pass must add no hits."""
    evidence = _LEAKING_SHAPES["bare_string"].format(s=POSITIONAL_SECRET)
    bandit = _bandit_finding(evidence)
    query, session = _query_for([_gitleaks_finding(), bandit], bandit)
    after_build = session.redaction_report().secrets_masked

    second = session.redact(query, None, None)
    assert second == query, "a second pass altered already-redacted text"
    assert session.redaction_report().secrets_masked == after_build, "secret double-counted"


def test_a_finding_without_a_location_still_falls_back_to_regex():
    """No file/line to anchor to - the patterns must still apply."""
    bandit = ScannerFinding(
        finding_id="b-nofile",
        scanner="bandit",
        category="hardcoded_password_string",
        severity=Severity.MEDIUM,
        rule_id="B105",
        message="m",
        raw_evidence=f'password = "{PASSWORD}"',
    )
    query, _ = _query_for([_gitleaks_finding(), bandit], bandit)
    assert PASSWORD not in query
    assert "REDACTED" in query


def test_outside_a_scan_session_the_regex_fallback_and_bounds_still_apply():
    """No session means no index - patterns only, with bounds and the gate intact."""
    from sentinelai.ai.repository_context import RepositoryContext
    from sentinelai.ai.retrieval.context_retrieval import _build_query

    bandit = _bandit_finding(f'api_key = "{AWS_KEY}"')
    query = _build_query(bandit, RepositoryContext())  # deliberately no scan_session
    assert AWS_KEY not in query, "the regex fallback failed outside a session"

    _, sink, _ = _embed([query])
    assert AWS_KEY not in sink["body"]

    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(AIEnrichmentError):
            make_ollama_embed_fn(host=HOST, model=MODEL)(["q"] * (_MAX_BATCH_TEXTS + 1))
    urlopen.assert_not_called()


# --------------------- 9, 10, 11: contracts and neighbouring protections

def test_operational_errors_are_exceptions_not_findings():
    """A bounds or egress failure raises; it never fabricates or mutates a finding."""
    with patch("urllib.request.urlopen"):
        with pytest.raises(AIEnrichmentError) as excinfo:
            make_ollama_embed_fn(host=HOST, model=MODEL)(["z"] * (_MAX_BATCH_TEXTS + 1))
    assert not isinstance(excinfo.value, ScannerFinding)
    assert isinstance(excinfo.value, Exception)


def test_local_endpoint_validation_is_still_enforced():
    """P1 must not have loosened the loopback restriction."""
    with pytest.raises(RemoteEndpointError):
        make_ollama_embed_fn(host="http://ollama.example.com:11434", model=MODEL)


def test_redirect_and_proxy_protections_are_still_installed():
    """Spot-check; the full behavioural proofs live in test_ai_ollama_redirects.py."""
    from sentinelai.ai import ollama_http

    handlers = ollama_http._NO_PROXY_NO_REDIRECT_OPENER.handlers
    assert any(isinstance(h, ollama_http._RefuseRedirects) for h in handlers)
    assert not [h for h in handlers if isinstance(h, urllib.request.ProxyHandler)]


def test_scanner_only_runs_never_embed(monkeypatch):
    """With no embedding model configured the retriever makes no request at all."""
    import sentinelai.ai.config as config_module
    from sentinelai.ai.factory import create_retriever_for_scan

    monkeypatch.setattr(config_module, "_settings", None)
    monkeypatch.delenv("SENTINELAI_AI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("SENTINELAI_AI_LLM_MODEL", raising=False)

    with patch("urllib.request.urlopen") as urlopen:
        retriever = create_retriever_for_scan()
        assert retriever.retrieve("anything", top_k=5) == []
    urlopen.assert_not_called()
