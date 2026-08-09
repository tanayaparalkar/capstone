"""
Centralized CLI exit codes.

A small, stable, documented set - one per *category* of outcome a CI/CD
pipeline or script needs to branch on, not one per individual error.
Every `typer.Exit(code=...)` in this codebase uses one of these; no raw
numeric literals are scattered through the CLI.

    SUCCESS (0)            The command completed and, for `scan`, no
                            --fail-on threshold was crossed (or none was
                            requested). This is the default even when
                            findings exist - SentinelAI is informational
                            by default; a CI pipeline opts into gating.

    SECURITY_FINDINGS (1)  The scan/report completed successfully, but
                            findings at or above the requested --fail-on
                            threshold were present. The tool worked
                            correctly - this is a security *policy*
                            outcome, not a tool failure.

    INVALID_INPUT (2)      The request can't be fulfilled as given: a
                            bad CLI argument/flag value, a repository
                            path that doesn't exist, a malformed or
                            unsupported saved report passed to `report`,
                            or an output path that can't be written.
                            Fixable by the user without any code change.

    PROVIDER_ERROR (3)     The FindingsProvider (scanner/backend
                            integration) raised while producing a scan
                            result. SentinelAI itself is working; the
                            thing it depends on is not.

    INTERNAL_ERROR (4)     An unexpected, unhandled failure inside
                            SentinelAI itself - a bug, not a policy or
                            input problem. Should be rare. Run with
                            --debug to see the full traceback.

Deliberately small: a CI pipeline needs to distinguish "the security
gate failed" from "SentinelAI couldn't even run" from "you passed
something invalid" - not enumerate every possible failure mode.
"""
from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    SECURITY_FINDINGS = 1
    INVALID_INPUT = 2
    PROVIDER_ERROR = 3
    INTERNAL_ERROR = 4
