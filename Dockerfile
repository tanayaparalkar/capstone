# SentinelAI - reproducible scanner image.
#
# This is a packaging and reproducibility layer, nothing more. The CLI
# inside behaves exactly as it does on a host: same scanners, same core
# and extended tiers, same correlation rules, same report contracts. No
# execution architecture is introduced here - there is no server, no
# daemon, no orchestrator, no database, and no queue in this image.
#
# What is deliberately NOT here:
#   - No Ollama server and no model weights. AI enrichment is opt-in and
#     needs a separately configured Ollama host - see the README. The
#     reproducible container path is scanner-only.
#   - No API keys, cloud credentials, or secret-bearing ARG/ENV values.
#   - No PostgreSQL, Redis, Qdrant, Chroma, Compose, or Kubernetes.
#   - No exposed ports: nothing in this image listens.
#
# Build (BuildKit, for TARGETARCH):
#   docker build -t sentinelai:local .
#
# Every externally downloaded binary is pinned by version and verified
# against the publishing project's own SHA-256 checksum file. See
# "Checksum trust" below for exactly what that does and does not prove.

FROM python:3.12-slim-bookworm

# Pinned, never "latest" - these are the exact versions this project's CI
# uses and that the Trivy/OSV wrappers' documented JSON shapes and
# exit-code behavior were verified against. Bump deliberately.
ARG GITLEAKS_VERSION=8.30.1
ARG TRIVY_VERSION=0.74.0
ARG OSV_SCANNER_VERSION=2.5.1

# Semgrep and Bandit come from PyPI rather than a release tarball, pinned
# to the versions measured in this project's benchmarks. pip verifies
# package hashes against PyPI's index metadata over TLS.
ARG SEMGREP_VERSION=1.172.0
ARG BANDIT_VERSION=1.9.4

# Supplied automatically by BuildKit. Asset names differ per architecture
# for all three downloaded tools, so this is required for the image to
# build on both x86_64 and Apple Silicon.
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ca-certificates: TLS for the downloads below and for OSV-Scanner's
# runtime queries to osv.dev.
# curl: fetches release archives to disk for checksum verification. It is
#   never piped into a shell.
# git: required at runtime, not optional. sentinelai/backend/git.py
#   deliberately lets GitCommandNotFound propagate, so scanning any git
#   repository without this binary fails with PROVIDER_ERROR.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- Downloaded scanner binaries -------------------------------------------
#
# Checksum trust: each tool publishes a SHA-256 checksum file as a release
# asset, fetched here from the same pinned tag and verified with
# `sha256sum -c` before anything is extracted or made executable. This
# reliably catches a truncated, corrupted, or mirror-mangled download.
# It is NOT an independent trust root - the checksum file ships from the
# same GitHub release as the binary it describes, so it cannot detect a
# compromised or re-tagged upstream release. Pinning digests directly in
# this file would be stronger, at the cost of six per-arch values needing
# a manual update on every version bump. No hash is hardcoded here, and
# none is invented.
#
# `curl -f` makes HTTP errors fail the build instead of silently writing
# an error page to disk. Each tool is installed in its own layer so a
# failure names the tool that failed.
#
# The matched checksum line is written to a file and `test -s` asserts it
# is non-empty before verification. This is not redundant: piping grep
# straight into `sha256sum -c -` was measured to exit 0 when grep matches
# nothing, so a renamed asset or a changed checksum-file format would skip
# verification silently and install an unverified binary. Failing on an
# empty match makes that impossible on any sha256sum implementation.

# GitLeaks - core tier.
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) gitleaks_arch='x64' ;; \
        arm64) gitleaks_arch='arm64' ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    archive="gitleaks_${GITLEAKS_VERSION}_linux_${gitleaks_arch}.tar.gz"; \
    base="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}"; \
    cd /tmp; \
    curl -fsSL -o "${archive}" "${base}/${archive}"; \
    curl -fsSL -o checksums.txt "${base}/gitleaks_${GITLEAKS_VERSION}_checksums.txt"; \
    grep " ${archive}\$" checksums.txt > expected.sha256; \
    test -s expected.sha256; \
    sha256sum -c expected.sha256; \
    tar -xzf "${archive}" gitleaks; \
    install -o root -g root -m 0755 gitleaks /usr/local/bin/gitleaks; \
    rm -f "${archive}" checksums.txt expected.sha256 gitleaks; \
    gitleaks version

# Trivy - extended tier.
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) trivy_arch='Linux-64bit' ;; \
        arm64) trivy_arch='Linux-ARM64' ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    archive="trivy_${TRIVY_VERSION}_${trivy_arch}.tar.gz"; \
    base="https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}"; \
    cd /tmp; \
    curl -fsSL -o "${archive}" "${base}/${archive}"; \
    curl -fsSL -o checksums.txt "${base}/trivy_${TRIVY_VERSION}_checksums.txt"; \
    grep " ${archive}\$" checksums.txt > expected.sha256; \
    test -s expected.sha256; \
    sha256sum -c expected.sha256; \
    tar -xzf "${archive}" trivy; \
    install -o root -g root -m 0755 trivy /usr/local/bin/trivy; \
    rm -f "${archive}" checksums.txt expected.sha256 trivy; \
    trivy --version

# OSV-Scanner - extended tier. Shipped as a bare binary, not an archive.
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) osv_arch='linux_amd64' ;; \
        arm64) osv_arch='linux_arm64' ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    binary="osv-scanner_${osv_arch}"; \
    base="https://github.com/google/osv-scanner/releases/download/v${OSV_SCANNER_VERSION}"; \
    cd /tmp; \
    curl -fsSL -o "${binary}" "${base}/${binary}"; \
    curl -fsSL -o SHA256SUMS "${base}/osv-scanner_SHA256SUMS"; \
    grep " ${binary}\$" SHA256SUMS > expected.sha256; \
    test -s expected.sha256; \
    sha256sum -c expected.sha256; \
    install -o root -g root -m 0755 "${binary}" /usr/local/bin/osv-scanner; \
    rm -f "${binary}" SHA256SUMS expected.sha256; \
    osv-scanner --version

# --- SentinelAI and its Python scanners ------------------------------------

# Semgrep and Bandit are the core tier's Python-installed scanners.
RUN pip install --no-cache-dir \
        "semgrep==${SEMGREP_VERSION}" \
        "bandit==${BANDIT_VERSION}"

# Only the CLI package is copied in - see .dockerignore for everything
# excluded from the build context (.git, caches, virtualenvs, generated
# reports, environment files, model data).
COPY sentinelai-cli/ /opt/sentinelai/sentinelai-cli/

# Installs sentinelai plus its declared runtime dependencies (typer, rich,
# pydantic, jinja2, GitPython) from pyproject.toml. The `sentinelai`
# console script lands on PATH and becomes the entrypoint below. Dev
# extras (pytest, pyyaml) are deliberately not installed: this is a
# runtime image, not a test runner.
RUN pip install --no-cache-dir /opt/sentinelai/sentinelai-cli

# --- Non-root runtime ------------------------------------------------------

# /workspace is the read-only mount point for the repository under scan.
# /output is a separate writable location, so scan results never have to
# be written into the mounted repository. It is created here and owned by
# the app user because Docker creates a missing host bind-mount directory
# as root, which an unprivileged container user could not then write to.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin sentinelai \
    && mkdir -p /workspace /output \
    && chown sentinelai:sentinelai /workspace /output

USER sentinelai

# A bind-mounted host repository is owned by a UID that does not exist in
# this image, and git refuses to operate on such a checkout ("detected
# dubious ownership"). backend/git.py does not catch that error, so
# without this the documented `docker run -v "$(pwd):/workspace:ro"`
# command fails with PROVIDER_ERROR on any git repository.
#
# This relaxes a protection aimed at a threat this container does not
# face: it is throwaway, unprivileged, and the mount is read-only. The
# tradeoff is stated in the README rather than left implicit.
RUN git config --global --add safe.directory '*'

WORKDIR /workspace

# `sentinelai` itself, so `docker run ... sentinelai:local scan /workspace`
# reads as the CLI it is. --help is the default because a bare
# `docker run sentinelai:local` should explain itself rather than scan an
# empty directory.
ENTRYPOINT ["sentinelai"]
CMD ["--help"]
