"""Tests for sentinelai.backend.dependencies - manifest dependency extraction."""
from pathlib import Path

import pytest

from sentinelai.backend import dependencies as dependencies_module
from sentinelai.backend.dependencies import Dependency, extract_dependencies
from sentinelai.backend.loader import load_repository


def _write(path: Path, content: str) -> None:
    path.write_text(content)


def test_empty_repository(tmp_path):
    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == []


def test_requirements_txt(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "requests==2.31.0\n"
        "flask>=2.0\n"
        "click<=8.0\n"
        "numpy~=1.26\n"
        "rich\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="click", version="<=8.0", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="flask", version=">=2.0", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="numpy", version="~=1.26", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="rich", version=None, ecosystem="pypi", source_file="requirements.txt"),
    ]


def test_pyproject_toml(tmp_path):
    _write(
        tmp_path / "pyproject.toml",
        '[project]\n'
        'name = "example"\n'
        'dependencies = [\n'
        '    "requests>=2.28.0",\n'
        '    "pydantic==2.6.0",\n'
        ']\n'
        '\n'
        '[project.optional-dependencies]\n'
        'dev = ["pytest>=8.0.0"]\n'
        '\n'
        '[build-system]\n'
        'requires = ["setuptools>=61.0"]\n'
        '\n'
        '[tool.some_tool]\n'
        'ignored = "value"\n',
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="pydantic", version="==2.6.0", ecosystem="pypi", source_file="pyproject.toml"),
        Dependency(name="requests", version=">=2.28.0", ecosystem="pypi", source_file="pyproject.toml"),
    ]


def test_both_files_together(tmp_path):
    _write(tmp_path / "requirements.txt", "flask==2.0\n")
    _write(tmp_path / "pyproject.toml", '[project]\ndependencies = ["requests>=2.0"]\n')

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="flask", version="==2.0", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="requests", version=">=2.0", ecosystem="pypi", source_file="pyproject.toml"),
    ]


def test_comments_are_ignored(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "# top-level comment\n"
        "requests==2.31.0  # inline comment\n"
        "   # indented comment\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt")]


def test_editable_installs_are_ignored(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "-e .\n"
        "-e git+https://example.com/pkg.git#egg=pkg\n"
        "requests==2.31.0\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt")]


def test_bare_url_dependencies_with_no_name_are_ignored(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "https://example.com/pkg.whl\n"
        "git+https://example.com/other.git\n"
        "requests==2.31.0\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt")]


def test_direct_references_are_partially_parsed_as_name_only(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "requests @ file:///tmp/requests.whl\n"
        "mypkg @ git+https://example.com/repo.git\n"
        "otherpkg @ https://example.com/pkg.whl\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="mypkg", version=None, ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="otherpkg", version=None, ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="requests", version=None, ecosystem="pypi", source_file="requirements.txt"),
    ]


def test_include_directives_and_pip_options_are_ignored(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "-r base.txt\n"
        "--index-url https://example.com/simple\n"
        "requests==2.31.0\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt")]


def test_duplicate_entries_are_removed(tmp_path):
    _write(tmp_path / "requirements.txt", "requests==2.31.0\nrequests==2.31.0\n")

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt")]


def test_deterministic_ordering_by_ecosystem_then_name(tmp_path):
    _write(tmp_path / "requirements.txt", "zeta==1.0\nalpha==1.0\nmu==1.0\n")

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert [dependency.name for dependency in result] == ["alpha", "mu", "zeta"]


def test_dependency_is_immutable(tmp_path):
    _write(tmp_path / "requirements.txt", "requests==2.31.0\n")
    result = extract_dependencies(load_repository(str(tmp_path)))

    with pytest.raises(Exception):
        result[0].version = "9.9.9"


def test_greater_less_and_not_equal_operators_are_supported(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "urllib3>2.0\n"
        "django<5.0\n"
        "cryptography!=3.3.2\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="cryptography", version="!=3.3.2", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="django", version="<5.0", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="urllib3", version=">2.0", ecosystem="pypi", source_file="requirements.txt"),
    ]


def test_arbitrary_equality_operator_is_supported(tmp_path):
    _write(tmp_path / "requirements.txt", "legacypkg===1.0.0.legacy\n")

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="legacypkg", version="===1.0.0.legacy", ecosystem="pypi", source_file="requirements.txt")
    ]


def test_extras_are_stripped_but_base_dependency_is_kept(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "requests[socks]\n"
        "uvicorn[standard]>=0.30\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="requests", version=None, ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="uvicorn", version=">=0.30", ecosystem="pypi", source_file="requirements.txt"),
    ]


def test_environment_markers_are_stripped_but_base_dependency_is_kept(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        'requests>=2.0; python_version >= "3.11"\n'
        'typing-extensions; python_version < "3.11"\n',
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="requests", version=">=2.0", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="typing-extensions", version=None, ecosystem="pypi", source_file="requirements.txt"),
    ]


def test_case_insensitive_duplicate_removal(tmp_path):
    _write(tmp_path / "requirements.txt", "Requests==2.31.0\nrequests==2.31.0\n")

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt")]


def test_same_package_in_both_files_is_kept_as_two_entries(tmp_path):
    _write(tmp_path / "requirements.txt", "requests==2.31.0\n")
    _write(tmp_path / "pyproject.toml", '[project]\ndependencies = ["requests>=2.0"]\n')

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="requests", version=">=2.0", ecosystem="pypi", source_file="pyproject.toml"),
    ]


def test_conflicting_version_specifiers_in_same_file_are_kept_as_two_entries(tmp_path):
    _write(tmp_path / "requirements.txt", "requests==2.31.0\nrequests>=2.0\n")

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="requests", version=">=2.0", ecosystem="pypi", source_file="requirements.txt"),
    ]


def test_tied_entries_have_deterministic_order_across_hash_seeds(tmp_path):
    # Regression guard for the set()-iteration-order issue: entries tied on
    # (ecosystem, name) must sort the same way regardless of string hash
    # randomization, which the sort key achieves by also covering version and
    # source_file. Verified empirically across PYTHONHASHSEED=1..5 before this
    # test was written; this test locks the specific, now-deterministic order in.
    _write(tmp_path / "requirements.txt", "requests==2.31.0\nrequests>=2.0\nrequests!=2.5.0\n")

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert [dependency.version for dependency in result] == ["!=2.5.0", "==2.31.0", ">=2.0"]


def test_whitespace_around_operator_does_not_affect_parsing(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "requests >=2.31\n"
        "flask>= 2.31\n"
        "click >= 2.31\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [
        Dependency(name="click", version=">=2.31", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="flask", version=">=2.31", ecosystem="pypi", source_file="requirements.txt"),
        Dependency(name="requests", version=">=2.31", ecosystem="pypi", source_file="requirements.txt"),
    ]


def test_malformed_lines_are_ignored(tmp_path):
    _write(
        tmp_path / "requirements.txt",
        "==\n"
        "requests>=\n"
        ">=2.0\n"
        "[]\n"
        "flask==2.0\n",
    )

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [Dependency(name="flask", version="==2.0", ecosystem="pypi", source_file="requirements.txt")]


def test_pep503_separator_variants_normalize_to_the_same_name(tmp_path):
    _write(tmp_path / "requirements.txt", "my_pkg==1.0\nmy-pkg==1.0\nmy.pkg==1.0\n")

    result = extract_dependencies(load_repository(str(tmp_path)))

    assert result == [Dependency(name="my-pkg", version="==1.0", ecosystem="pypi", source_file="requirements.txt")]


def test_invalid_pyproject_toml_propagates_uncaught(tmp_path):
    _write(tmp_path / "pyproject.toml", "[project\ndependencies = [")

    with pytest.raises(dependencies_module.tomllib.TOMLDecodeError):
        extract_dependencies(load_repository(str(tmp_path)))
