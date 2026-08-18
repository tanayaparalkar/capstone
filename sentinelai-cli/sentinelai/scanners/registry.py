"""
Scanner registry - an explicit, instantiable collection of named Scanner
instances.

Deliberately not a module-level singleton/global dict: a global registry
would leak state between tests (registering a scanner in one test would
still be visible in the next) and would make "which scanners are active"
implicit rather than something a caller explicitly constructs and passes
around. ScannerRegistry is instantiated per use instead.

No automatic discovery (e.g. scanning sentinelai/scanners/ for Scanner
subclasses on import): every registration is an explicit register() call,
so it's always obvious from reading the calling code which scanners are
active. No consumer has asked for plugin-style discovery, and building it
would need filesystem/import-machinery knowledge this framework
deliberately does not have.

Each registration also carries a ScannerTier, which is the only thing
that decides whether a scanner participates in a default scan. It lives
here rather than in the orchestrator (which deliberately knows nothing
about registries) or in each scanner class (which would make a tool's
tier a property of the tool rather than of this project's policy about
it). Both `register(name, scanner)` and `list_scanners()` keep their
previous signatures and their previous behavior, so every existing caller
is unaffected.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from sentinelai.contracts import ScannerTier

from .base import Scanner
from .exceptions import ConfigurationError


@dataclass(frozen=True)
class ScannerRegistration:
    name: str
    scanner: Scanner
    # Which scanner set this registration belongs to. Defaults to CORE so
    # constructing a ScannerRegistration positionally (as every existing
    # caller and test does) keeps working unchanged.
    tier: ScannerTier = ScannerTier.CORE


class ScannerRegistry:
    """A named collection of Scanner instances, registered explicitly by the caller."""

    def __init__(self) -> None:
        self._scanners: Dict[str, Scanner] = {}
        self._tiers: Dict[str, ScannerTier] = {}

    def register(self, name: str, scanner: Scanner, tier: ScannerTier = ScannerTier.CORE) -> None:
        """Register `scanner` under `name` in `tier`. Raises ConfigurationError if `name` is already registered.

        `tier` defaults to CORE, which is what keeps this signature
        backward-compatible: every pre-existing two-argument register() call
        registers into the core set and behaves exactly as before.
        """
        if name in self._scanners:
            raise ConfigurationError(f"A scanner named '{name}' is already registered")
        self._scanners[name] = scanner
        self._tiers[name] = tier

    def get(self, name: str) -> Scanner:
        """Return the scanner registered under `name`. Raises ConfigurationError if none is registered."""
        try:
            return self._scanners[name]
        except KeyError:
            raise ConfigurationError(f"No scanner named '{name}' is registered") from None

    def list_scanners(self, tier: Optional[ScannerTier] = None) -> Tuple[ScannerRegistration, ...]:
        """Return registrations sorted by name, so ordering never depends on registration order.

        With no argument, every registration is returned - the pre-existing
        behavior, unchanged. `tier=CORE` returns only the core scanners;
        `tier=EXTENDED` returns the core scanners *plus* the extended ones,
        because extended is a superset of core rather than a separate set:
        adding dependency scanning must never take code scanning away.
        """
        return tuple(
            ScannerRegistration(name=name, scanner=self._scanners[name], tier=self._tiers[name])
            for name in sorted(self._scanners)
            if tier is None or tier is ScannerTier.EXTENDED or self._tiers[name] is ScannerTier.CORE
        )
