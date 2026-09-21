"""
Timing instrumentation for pipeline stages.

Exists because eight stages need the same three lines - read the clock, run
the work, log how long it took - and eight copies of that would drift in
format, in level, and in whether they still log when the stage raises.

Deliberately small. It adds no state, no configuration, no handler and no
logger of its own: records go to the same "sentinelai" logger every other
module already uses, so whatever a caller has configured for that logger
applies here unchanged.

Observation only, never control flow. The context manager re-raises anything
the body raises, having logged the elapsed time first, so a stage that fails is
still measured and the failure reaches its existing handler untouched. Nothing
here catches, retries, suppresses or alters an exception, and no caller's
behaviour depends on whether a record was emitted.

Levels follow the project's existing convention. Stage boundaries are INFO and
per-item detail is DEBUG; nothing here emits WARNING or ERROR, because no
handler is attached by default and logging.lastResort would turn those into new
stderr output - a behaviour change this module must not cause.
"""
import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("sentinelai")


def format_fields(**fields: object) -> str:
    """Render key=value pairs in the order given, matching the existing message style."""
    return " ".join(f"{key}={value}" for key, value in fields.items())


@contextmanager
def log_duration(stage: str, level: int = logging.INFO, **fields: object) -> Iterator[dict]:
    """Log how long `stage` took, whether it succeeds or raises.

    Yields a mutable dict the body may add fields to - a stage usually does not
    know its own output counts until it has finished, and this avoids a second
    log call just to report them.

    On failure the duration is still logged, at the same level, with
    `outcome=failed`. The exception then propagates unchanged; this deliberately
    does not log at ERROR, because the existing handlers at the call sites
    already do that and duplicating it would put new records on stderr.
    """
    extra: dict = {}
    started = time.monotonic()
    try:
        yield extra
    except BaseException:
        logger.log(
            level,
            "%s: failed after %.3fs%s",
            stage,
            time.monotonic() - started,
            _suffix(fields, extra, outcome="failed"),
        )
        raise
    logger.log(
        level,
        "%s: completed in %.3fs%s",
        stage,
        time.monotonic() - started,
        _suffix(fields, extra),
    )


def _suffix(fields: dict, extra: dict, **overrides: object) -> str:
    merged = {**fields, **extra, **overrides}
    return f" {format_fields(**merged)}" if merged else ""
