"""
Pure query functions over an already-loaded knowledge base.

Takes list[KnowledgeBaseEntry] as an explicit parameter rather than
loading it - loading is loader.py's job (I/O); this is pure in-memory
filtering, the same split models.py/loader.py already makes one level
up. No dependency on loader.py: these functions don't care where the
list came from.

Naming follows core/severity.py's filter_by_<criterion> convention
(filter_by_severity) rather than inventing a different verb for the
same idiom.

Documented assumption about the corpus (per project owner's request:
document, don't silently encode into the data or the code): nothing in
KnowledgeBaseEntry's schema guarantees category or cwe is unique across
entries. The bundled fixture happens to have one entry per category
today, but that is a property of the current data, not a contract this
module enforces or relies on - both functions return every match, in
the order they appear in the input list, not just the first.
"""
from .models import KnowledgeBaseEntry


def filter_by_category(entries: list[KnowledgeBaseEntry], category: str) -> list[KnowledgeBaseEntry]:
    """Return every entry whose category exactly matches `category`, in input order. Empty list on no match."""
    return [e for e in entries if e.category == category]


def filter_by_cwe(entries: list[KnowledgeBaseEntry], cwe: str) -> list[KnowledgeBaseEntry]:
    """Return every entry whose cwe exactly matches `cwe`, in input order. Empty list on no match."""
    return [e for e in entries if e.cwe == cwe]
