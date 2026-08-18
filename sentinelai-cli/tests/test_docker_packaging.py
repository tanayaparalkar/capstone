"""
Static validation of the Docker packaging layer (Dockerfile, .dockerignore,
and the README instructions that describe them).

Deliberately lightweight and daemon-free, exactly like
tests/test_ci_workflow.py: these parse and assert on file *content*, they
never invoke `docker`, never build an image, and never reach the network.
They therefore run in the default `pytest -q` suite in milliseconds and
need no marker. What they can catch is "this file is malformed" or "this
file no longer does what the README claims" - which is most of what goes
wrong with a Dockerfile between builds.

What they explicitly cannot catch is whether the image actually builds or
whether the pinned URLs still resolve. That needs a real build, which is
an opt-in manual step documented in the README.

The negative assertions (no secrets, no Ollama pull, no infrastructure,
no privileged mode) matter more than the positive ones. A missing tool
fails loudly on first use; a credential or a database baked into a
published image fails silently and permanently.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"
README_PATH = REPO_ROOT / "sentinelai-cli" / "README.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _instructions(dockerfile: str) -> str:
    """The Dockerfile with comment lines stripped.

    Every negative assertion below runs against this rather than the raw
    text: the file explains at length what it deliberately does *not*
    contain ("No PostgreSQL, Redis, Qdrant..."), and matching those
    sentences would make the guards pass or fail on prose instead of on
    instructions.
    """
    return "\n".join(
        line for line in dockerfile.splitlines() if line.strip() and not line.strip().startswith("#")
    )


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return _text(DOCKERFILE_PATH)


@pytest.fixture(scope="module")
def instructions(dockerfile) -> str:
    return _instructions(dockerfile)


@pytest.fixture(scope="module")
def dockerignore() -> str:
    return _text(DOCKERIGNORE_PATH)


@pytest.fixture(scope="module")
def readme() -> str:
    return _text(README_PATH)


# --- 1-2. the files exist --------------------------------------------------------------------------------


def test_dockerfile_exists():
    assert DOCKERFILE_PATH.is_file(), f"no Dockerfile at {DOCKERFILE_PATH}"


def test_dockerignore_exists():
    assert DOCKERIGNORE_PATH.is_file(), f"no .dockerignore at {DOCKERIGNORE_PATH}"


# --- 3. pinned base image --------------------------------------------------------------------------------


def test_base_image_is_explicitly_tagged(instructions):
    """An untagged or `:latest` base makes the image unreproducible by definition."""
    from_lines = [line for line in instructions.splitlines() if line.upper().startswith("FROM ")]

    assert len(from_lines) == 1, f"expected exactly one FROM, got {from_lines}"
    image = from_lines[0].split()[1]
    assert ":" in image, f"base image '{image}' has no tag"
    assert not image.endswith(":latest"), "base image must not be :latest"
    assert image.startswith("python:3.12"), f"base image '{image}' is not the expected Python 3.12 line"


def test_base_image_python_satisfies_requires_python():
    """The image's Python must satisfy pyproject.toml's requires-python."""
    pyproject = _text(REPO_ROOT / "sentinelai-cli" / "pyproject.toml")
    minimum = re.search(r'requires-python\s*=\s*"[><=]*(\d+)\.(\d+)"', pyproject)

    assert minimum, "could not read requires-python from pyproject.toml"
    major, minor = int(minimum.group(1)), int(minimum.group(2))
    image_version = re.search(r"FROM python:(\d+)\.(\d+)", _text(DOCKERFILE_PATH))
    assert image_version, "could not read the Python version from the base image"
    assert (int(image_version.group(1)), int(image_version.group(2))) >= (major, minor)


# --- 4. all five scanners installed ----------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["semgrep", "bandit", "gitleaks", "trivy", "osv-scanner"])
def test_every_scanner_is_installed(instructions, tool):
    assert tool in instructions, f"{tool} is never installed in the image"


@pytest.mark.parametrize(
    "tool,variable",
    [
        ("gitleaks", "GITLEAKS_VERSION"),
        ("trivy", "TRIVY_VERSION"),
        ("osv-scanner", "OSV_SCANNER_VERSION"),
        ("semgrep", "SEMGREP_VERSION"),
        ("bandit", "BANDIT_VERSION"),
    ],
)
def test_every_scanner_version_is_pinned_by_an_arg(instructions, tool, variable):
    """Pinned to an explicit version, never floating - CI-style determinism."""
    declaration = re.search(rf"^ARG {variable}=(\S+)", instructions, re.MULTILINE)

    assert declaration, f"{tool} has no pinned {variable} ARG"
    value = declaration.group(1)
    assert value not in {"latest", ""}, f"{variable} must be an explicit version, got '{value}'"
    assert re.match(r"^\d+\.\d+", value), f"{variable}='{value}' is not a version number"


def test_downloaded_archives_are_checksum_verified(instructions):
    """Every downloaded binary is verified before it is installed.

    The checksum files ship from the same release as the binaries, so this
    proves integrity of the download, not authenticity of the release - a
    limitation stated in the Dockerfile and README rather than papered over.
    """
    assert instructions.count("sha256sum -c") >= 3, "expected a checksum check per downloaded tool"
    for tool in ("gitleaks", "trivy", "osv-scanner"):
        assert re.search(rf"{tool}.*\n(?:.*\n)*?.*sha256sum -c", instructions), f"{tool} download is unverified"


def test_checksum_verification_cannot_be_skipped_by_an_empty_match(instructions):
    """A grep that matches nothing must fail the build, not silently skip verification.

    Measured directly: `grep ... | sha256sum -c -` exits 0 when grep matches
    nothing, because the pipeline's status is sha256sum's and some
    implementations accept empty input. A renamed release asset or a changed
    checksum-file format would then install an unverified binary. Each
    download therefore writes the matched line to a file and asserts it is
    non-empty before verifying.
    """
    assert "test -s expected.sha256" in instructions, "an empty checksum match would skip verification"
    assert instructions.count("test -s expected.sha256") >= 3, "not every download guards against an empty match"
    assert not re.search(r"grep[^\n]*\|\s*sha256sum", instructions), (
        "piping grep directly into sha256sum silently passes on an empty match"
    )


def test_no_hardcoded_checksums_are_invented(instructions):
    """Hashes come from each project's published checksum file, never from this repo.

    A hand-copied digest silently stops matching on the next version bump,
    and an invented one is worse than none at all.
    """
    assert not re.search(r"\b[0-9a-f]{64}\b", instructions), "a raw sha256 digest is hardcoded in the Dockerfile"


def test_downloads_are_never_piped_into_a_shell(instructions):
    """`curl | sh` executes whatever the endpoint returns, unverified."""
    assert not re.search(r"(curl|wget)[^\n|]*\|\s*(ba|z|)sh", instructions)


def test_package_installation_is_non_interactive(instructions):
    assert "DEBIAN_FRONTEND=noninteractive" in instructions
    assert re.search(r"apt-get install[^\n]*(--yes|-y)\b", instructions), "apt-get install is not non-interactive"


def test_git_is_installed_because_the_backend_requires_it(instructions):
    """backend/git.py lets GitCommandNotFound propagate: no git binary means PROVIDER_ERROR."""
    assert re.search(r"apt-get install(?:.|\n)*?\bgit\b", instructions)


# --- 5. non-root runtime ---------------------------------------------------------------------------------


def test_a_non_root_user_is_created_and_selected(instructions):
    assert re.search(r"useradd[^\n]*sentinelai", instructions), "no application user is created"
    user_lines = [line.strip() for line in instructions.splitlines() if line.strip().upper().startswith("USER ")]
    assert user_lines, "the image never switches away from root"
    assert user_lines[-1].split()[1] != "root", "the image runs as root at runtime"


def test_the_user_switch_happens_before_the_entrypoint(instructions):
    """A USER after ENTRYPOINT would still run the container as root."""
    last_user = instructions.upper().rfind("\nUSER ")
    entrypoint = instructions.upper().rfind("\nENTRYPOINT")

    assert last_user != -1 and entrypoint != -1
    assert last_user < entrypoint, "USER must precede ENTRYPOINT"


# --- 6. working directory --------------------------------------------------------------------------------


def test_workdir_is_the_documented_workspace(instructions):
    workdirs = re.findall(r"^WORKDIR\s+(\S+)", instructions, re.MULTILINE)

    assert workdirs, "no WORKDIR is set"
    assert workdirs[-1] == "/workspace"


def test_a_separate_writable_output_directory_exists(instructions):
    """Scan output must never be written into the read-only mounted repository."""
    assert "/output" in instructions
    assert re.search(r"chown[^\n]*sentinelai[^\n]*/output|chown[^\n]*/output", instructions), (
        "/output is not owned by the application user, so a non-root process could not write to it"
    )


# --- 7. entrypoint ---------------------------------------------------------------------------------------


def test_entrypoint_invokes_sentinelai(instructions):
    entrypoint = re.search(r'^ENTRYPOINT\s+(\[.*\]|.*)$', instructions, re.MULTILINE)

    assert entrypoint, "no ENTRYPOINT is defined"
    assert "sentinelai" in entrypoint.group(1)


def test_entrypoint_uses_exec_form(instructions):
    """Shell form would wrap the CLI in /bin/sh, breaking signal handling and argv passthrough."""
    entrypoint = re.search(r"^ENTRYPOINT\s+(.*)$", instructions, re.MULTILINE).group(1).strip()

    assert entrypoint.startswith("["), f"ENTRYPOINT must use exec form, got {entrypoint}"


def test_the_documented_run_commands_reach_the_cli(instructions):
    """`docker run ... sentinelai:local scan /workspace ...` must append to the entrypoint, not replace it."""
    assert re.search(r'ENTRYPOINT\s+\["sentinelai"\]', instructions), (
        "the entrypoint must be exactly `sentinelai` so documented `scan ...` arguments append to it"
    )


# --- 8. nothing that does not belong in the image --------------------------------------------------------


@pytest.mark.parametrize(
    "label,pattern",
    [
        ("an API key", r"API_KEY|APIKEY"),
        ("a secret", r"\bSECRET\b|SECRET_KEY|ACCESS_TOKEN"),
        ("AWS credentials", r"AWS_ACCESS_KEY|AWS_SECRET|SECRET_ACCESS_KEY"),
        ("a cloud LLM credential", r"OPENAI|ANTHROPIC_API|GEMINI|HUGGINGFACE"),
        ("an Ollama model pull", r"ollama\s+(pull|serve|run)"),
        ("a bundled model file", r"\.gguf|\.safetensors"),
        ("PostgreSQL", r"postgres|psql|POSTGRES"),
        ("Qdrant", r"qdrant"),
        ("Chroma", r"chromadb|chroma-"),
        ("Redis", r"\bredis\b"),
        ("Docker-in-Docker", r"docker\.sock|docker:dind|docker-in-docker"),
        ("privileged mode", r"--privileged|privileged:\s*true"),
        ("Kubernetes", r"kubectl|kubernetes"),
        ("Docker Compose", r"docker-compose|docker\s+compose"),
    ],
)
def test_dockerfile_contains_no(instructions, label, pattern):
    match = re.search(pattern, instructions, re.IGNORECASE)

    assert match is None, f"Dockerfile contains {label}: {match.group(0)!r}"


def test_no_ports_are_exposed(instructions):
    """Nothing in this image listens; an EXPOSE would misrepresent that."""
    assert not re.search(r"^EXPOSE\b", instructions, re.MULTILINE)


def test_env_and_arg_values_carry_no_secrets(instructions):
    """Every ARG/ENV value must be a version, a path, or a build flag - never a credential."""
    allowed = re.compile(
        r"^(PYTHON\w*|PIP_\w+|DEBIAN_FRONTEND|TARGETARCH|\w*VERSION)$"
    )
    for line in instructions.splitlines():
        match = re.match(r"^(?:ARG|ENV)\s+([A-Z_][A-Z0-9_]*)=?", line.strip())
        if match:
            name = match.group(1)
            assert allowed.match(name), f"unexpected build variable '{name}' - is it a secret?"


def test_no_compose_file_was_added():
    """Compose was explicitly out of scope for this phase."""
    for candidate in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        assert not (REPO_ROOT / candidate).exists(), f"{candidate} was not approved for this phase"


# --- 9. .dockerignore coverage ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,pattern",
    [
        ("git metadata", ".git"),
        ("Python bytecode caches", "**/__pycache__/"),
        ("compiled Python files", "**/*.py[cod]"),
        ("virtual environments", "**/.venv/"),
        ("pytest caches", "**/.pytest_cache/"),
        ("environment files", "**/.env"),
        ("private keys", "**/*.pem"),
        ("model weights", "**/*.gguf"),
        ("model directories", "**/models/"),
        ("generated JSON reports", "**/scan-result*.json"),
        ("generated SARIF reports", "**/*.sarif"),
        ("generated HTML reports", "**/sentinelai-report*.html"),
        ("the benchmark fixture", "sentinelai-manual-test/"),
    ],
)
def test_dockerignore_excludes(dockerignore, label, pattern):
    entries = {line.strip() for line in dockerignore.splitlines() if line.strip() and not line.startswith("#")}

    assert pattern in entries, f"{label} is not excluded from the build context ('{pattern}' missing)"


def test_recursive_patterns_use_a_double_star_prefix(dockerignore):
    """.dockerignore patterns are root-anchored and do not recurse on their own.

    A bare `__pycache__/` would exclude only /__pycache__ and would leave
    sentinelai-cli/sentinelai/__pycache__/ in the build context.
    """
    must_recurse = ("__pycache__", ".pytest_cache", ".venv", ".env", ".DS_Store")
    entries = [line.strip() for line in dockerignore.splitlines() if line.strip() and not line.startswith("#")]

    for name in must_recurse:
        matching = [e for e in entries if name in e]
        assert matching, f"no rule mentions {name}"
        assert all(e.startswith("**/") for e in matching), (
            f"{name} rules must start with '**/' to match at any depth: {matching}"
        )


def test_no_negation_reintroduces_an_excluded_secret(dockerignore):
    """A `!` rule can silently undo an exclusion above it."""
    negations = [line.strip() for line in dockerignore.splitlines() if line.strip().startswith("!")]

    assert negations == [], f"unexpected re-inclusion rules: {negations}"


# --- 10-11. README documents both modes and the AI caveat ------------------------------------------------


def test_readme_documents_the_core_docker_command(readme):
    assert "docker run --rm" in readme
    assert '-v "$(pwd):/workspace:ro"' in readme
    assert "scan /workspace --format json --output /output/scan-result.json" in readme


def test_readme_documents_the_extended_docker_command(readme):
    assert "scan /workspace --extended --format json --output /output/scan-result.json" in readme


def test_readme_documents_the_build_command(readme):
    assert "docker build" in readme
    assert "sentinelai:local" in readme


def test_readme_mounts_the_repository_read_only(readme):
    """The scanned repository must never be writable from the container."""
    docker_lines = [line for line in readme.splitlines() if "/workspace" in line and "-v " in line]

    assert docker_lines, "no documented volume mount for /workspace"
    assert all(":ro" in line for line in docker_lines), (
        f"every /workspace mount must be read-only: {docker_lines}"
    )


def test_readme_writes_output_outside_the_scanned_repository(readme):
    assert "/output" in readme
    assert "--output /output/" in readme


def test_readme_states_ai_enrichment_is_optional_and_unbundled(readme):
    """No claim may imply the image can do AI enrichment on its own."""
    lowered = readme.lower()

    assert "optional" in lowered
    assert "ollama" in lowered
    assert re.search(r"no (ollama )?(model|model weights|models) (are |is )?(not )?", lowered) or (
        "not bundled" in lowered or "not included" in lowered
    ), "README must state that model weights are not bundled"
    assert "scanner-only" in lowered


def test_the_documented_output_directory_is_gitignored(readme):
    """The README tells users to `mkdir -p sentinelai-output` inside the repo they scan.

    Running the documented command therefore leaves generated reports in the
    working tree of whatever repository was scanned - including this one. They
    are reproducible output, not source, so .gitignore must cover them or they
    will be offered up for commit after every container run.
    """
    assert "mkdir -p sentinelai-output" in readme
    gitignore = _text(REPO_ROOT / ".gitignore")
    entries = {line.strip() for line in gitignore.splitlines() if line.strip() and not line.startswith("#")}

    assert "sentinelai-output/" in entries, "the documented Docker output directory is not gitignored"


def test_readme_documents_the_checksum_limitation(readme):
    """The honest bound on what checksum verification proves."""
    lowered = readme.lower()

    assert "checksum" in lowered
    assert "same release" in lowered or "not an independent trust root" in lowered
