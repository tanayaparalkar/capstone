"""
Security knowledge base entry schema.

KnowledgeBaseEntry is the shape of one entry in the corpus
ai/retrieval/ retrieves from - static, per-vulnerability-type facts
that exist independently of any particular scan. Kept separate from
loader.py (I/O that constructs instances) and repository.py (query
functions over already-loaded instances), the same way
contracts/scanner_finding.py is separate from providers/mock_provider.py:
schema and behavior are different files even when tightly coupled.

Fields map directly to either a named consumer already in the frozen
architecture or an already-frozen field on contracts.AIEnrichedFinding
that it supplies source material for - nothing here is speculative:

    id             -> referenced by ai/retrieval/models.py's
                       RetrievedChunk (source KB entry id)
    category, cwe  -> looked up by security_kb/repository.py, and
                       correlated against ScannerFinding.category /
                       ScannerFinding.cwe
    vulnerable_pattern, exploit_condition, remediation_guidance
                   -> source material for AIEnrichedFinding.explanation /
                      .exploit_path / .remediation respectively
    references     -> source material for AIEnrichedFinding.references

category is a free-form str, not an enum, for the same reason
ScannerFinding.category is: it must match Nithanth's open-ended
taxonomy without a second enum to keep in sync (see docs/CONTRACTS.md).
cwe follows ScannerFinding.cwe's shape exactly (Optional[str], no
format constraint).
"""
from typing import Optional

from pydantic import BaseModel, Field


class KnowledgeBaseEntry(BaseModel):
    id: str = Field(..., description="Stable identifier for this knowledge-base entry, e.g. 'KB-SQLI-001'.")
    category: str = Field(
        ..., description="Vulnerability category, matching ScannerFinding.category, e.g. 'sql-injection'."
    )
    cwe: Optional[str] = Field(None, description="CWE identifier, e.g. 'CWE-89', where applicable.")
    vulnerable_pattern: str = Field(
        ..., description="Description of the vulnerable code pattern this entry covers."
    )
    exploit_condition: Optional[str] = Field(
        None, description="Narrative of the conditions under which this pattern is exploitable, if applicable."
    )
    remediation_guidance: str = Field(..., description="Standard remediation guidance for this vulnerability type.")
    references: list[str] = Field(
        default_factory=list, description="External references, e.g. CWE/OWASP links."
    )
