"""
Dependency extraction - declared package dependencies read from manifest
files, with no judgment about whether any of them are vulnerable.

Vulnerability lookups (OSV, Trivy, ...) are a scanner concern and belong
to a later module that consumes the plain list this one produces; mixing
that in here would couple a static, offline read of a manifest file to a
network call, which is a different failure mode entirely.

MVP scope is requirements.txt and pyproject.toml's [project.dependencies]
only - the two formats needed for this project's own ecosystem. Other
manifests (package.json, pom.xml, go.mod, lockfiles, ...) are a future
module's concern, not a reason to add speculative parsing here now.

Design decisions worth stating explicitly, since each was a place a
plain reading of "just parse the requirement string" goes wrong:

- Extras ("uvicorn[standard]") are stripped and discarded, not preserved.
  A future vulnerability-matching consumer keys off (ecosystem, name,
  version) - extras select optional code paths, not a different
  package or version, so they carry no information that consumer needs.
  Discarding them still requires capturing the base name and version
  correctly, which the parser now does.
- Environment markers ("; python_version >= \"3.11\"") are stripped and
  discarded, not preserved or rejected. They describe *when* pip would
  install a dependency, not what it is; this module extracts declared
  identity, not environment resolution. The base name and version before
  the ";" are still captured.
- Invalid pyproject.toml is not caught here: tomllib.TOMLDecodeError
  propagates to the caller unwrapped. A malformed manifest is a genuine
  defect in the scanned repository, not a normal/expected state (unlike
  "no pyproject.toml at all", which is handled by simply not finding the
  file) - swallowing it would silently under-report dependencies, which
  is the wrong failure mode for a security tool.
- Duplicate removal is exact-tuple equality only: the same package
  declared in requirements.txt and pyproject.toml is kept as two entries
  (different source_file), and the same package listed twice in one file
  with different version specifiers is also kept as two entries (they
  are different declarations, not duplicates) - only a byte-identical
  redeclaration collapses. Name comparison for equality is
  case-insensitive by construction, since normalization lowercases
  before the Dependency is built.
- PEP 508 direct references ("name @ target", e.g. "pkg @ https://...",
  "pkg @ git+https://...", "pkg @ file:///...") are partially parsed:
  the name is kept, version is always None. The part after "@" is a
  source location, not a version specifier, and there is no meaningful
  version to extract from it - but the package is still a genuine
  declared dependency, so dropping it entirely (as a bare, nameless URL
  line must be) would under-report. A bare URL/VCS line with no "name @"
  prefix has no name to recover and is still skipped outright.
- A version operator with no version value ("requests>=") is treated as
  a malformed specifier and the whole line is dropped, rather than
  guessing that an unconstrained version was intended - the input is
  broken, and silently inventing a meaning for it would be worse than
  reporting nothing.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .loader import LoadedRepository

# Sorted longest-first by construction (not by manually-maintained order) so a
# two-character operator can never be mistaken for a one-character prefix of
# itself (">" vs ">=") and "===" is matched before "==" - correctness doesn't
# depend on remembering to list entries in the right order.
_VERSION_OPERATORS = tuple(sorted(("==", "!=", ">=", "<=", "~=", "===", ">", "<"), key=len, reverse=True))
_EXTRAS_PATTERN = re.compile(r"\[[^\]]*\]")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")
_NAME_NORMALIZATION_PATTERN = re.compile(r"[-_.]+")


@dataclass(frozen=True)
class Dependency:
    name: str
    version: Optional[str]
    ecosystem: str
    source_file: str


def extract_dependencies(repository: LoadedRepository) -> List[Dependency]:
    """Read requirements.txt and pyproject.toml (if present) under `repository.absolute_path`."""
    dependencies: List[Dependency] = []

    requirements_path = repository.absolute_path / "requirements.txt"
    if requirements_path.is_file():
        dependencies.extend(_read_requirements_txt(requirements_path, repository.absolute_path))

    pyproject_path = repository.absolute_path / "pyproject.toml"
    if pyproject_path.is_file():
        dependencies.extend(_read_pyproject_toml(pyproject_path, repository.absolute_path))

    # Sort key covers every field, not just (ecosystem, name): set() iteration order
    # depends on string hashing, which is randomized per-process, so entries tied on
    # (ecosystem, name) - e.g. the same package pinned differently in two files -
    # would otherwise come out in a different relative order on every run.
    unique_dependencies = sorted(
        set(dependencies),
        key=lambda dependency: (dependency.ecosystem, dependency.name, dependency.version or "", dependency.source_file),
    )
    return unique_dependencies


def _read_requirements_txt(path: Path, repository_root: Path) -> List[Dependency]:
    source_file = str(path.relative_to(repository_root))
    dependencies = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):
            # Pip options and directives: -e (editable), -r (include), --index-url, etc.
            continue
        if "://" in line and "@" not in line:
            # Bare URL / VCS reference with no name to recover (e.g. "git+https://...").
            # "name @ target" lines fall through to _parse_requirement, which extracts
            # the name and drops the target - see module docstring.
            continue

        parsed = _parse_requirement(line)
        if parsed is None:
            continue
        name, version = parsed
        dependencies.append(Dependency(name=name, version=version, ecosystem="pypi", source_file=source_file))
    return dependencies


def _read_pyproject_toml(path: Path, repository_root: Path) -> List[Dependency]:
    source_file = str(path.relative_to(repository_root))
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    raw_dependencies = data.get("project", {}).get("dependencies", [])
    dependencies = []
    for raw_dependency in raw_dependencies:
        if not isinstance(raw_dependency, str):
            continue
        parsed = _parse_requirement(raw_dependency)
        if parsed is None:
            continue
        name, version = parsed
        dependencies.append(Dependency(name=name, version=version, ecosystem="pypi", source_file=source_file))
    return dependencies


def _parse_requirement(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """Split a requirement string into (normalized name, version specifier or None)."""
    text = text.split(";", 1)[0]  # environment marker - see module docstring
    text = _EXTRAS_PATTERN.sub("", text, count=1)  # extras - see module docstring
    text = text.strip()
    if not text:
        return None

    if "@" in text:
        # PEP 508 direct reference ("name @ target") - see module docstring.
        name = text.split("@", 1)[0].strip()
        if not _NAME_PATTERN.match(name):
            return None
        return _normalize_name(name), None

    for operator in _VERSION_OPERATORS:
        if operator in text:
            name_part, _, version_part = text.partition(operator)
            name = name_part.strip()
            version_value = version_part.strip()
            if not version_value:
                # Operator with no version value ("requests>=") - see module docstring.
                return None
            version = operator + version_value
            break
    else:
        name = text
        version = None

    if not _NAME_PATTERN.match(name):
        return None
    return _normalize_name(name), version


def _normalize_name(name: str) -> str:
    """PEP 503 normalization: runs of -._ collapse to a single '-', lowercased."""
    return _NAME_NORMALIZATION_PATTERN.sub("-", name).strip().lower()
