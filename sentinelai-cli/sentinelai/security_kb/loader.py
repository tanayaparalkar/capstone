"""
Loads the bundled security knowledge base from its JSON fixture.

One job: read the file, construct KnowledgeBaseEntry instances, return
them. No caching (unlike ai/config.py's get_settings(), there is no
invariant here that makes re-reading wrong - a second call re-reading
the file is simply correct), no class (there is no ABC to implement,
unlike providers/mock_provider.py's MockFindingsProvider), no logging,
no retry, no custom exception wrapping. FileNotFoundError,
json.JSONDecodeError, and pydantic.ValidationError are left to propagate
as themselves - there is no CLI caller here needing them collapsed into
one type the way reporting/loader.py's ReportLoadError does for
sentinelai.main's `report` command.

DEFAULT_KB_DATA_PATH follows providers/mock_provider.py's
DEFAULT_MOCK_DATA_PATH pattern: a path constant next to the code that
owns it, overridable per call for tests.
"""
import json
from pathlib import Path

from .models import KnowledgeBaseEntry

DEFAULT_KB_DATA_PATH = Path(__file__).parent / "data" / "knowledge_base.json"


def load_knowledge_base(path: Path = DEFAULT_KB_DATA_PATH) -> list[KnowledgeBaseEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [KnowledgeBaseEntry.model_validate(item) for item in raw]
