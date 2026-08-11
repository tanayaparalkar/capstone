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
"""
from dataclasses import dataclass
from typing import Dict, Tuple

from .base import Scanner
from .exceptions import ConfigurationError


@dataclass(frozen=True)
class ScannerRegistration:
    name: str
    scanner: Scanner


class ScannerRegistry:
    """A named collection of Scanner instances, registered explicitly by the caller."""

    def __init__(self) -> None:
        self._scanners: Dict[str, Scanner] = {}

    def register(self, name: str, scanner: Scanner) -> None:
        """Register `scanner` under `name`. Raises ConfigurationError if `name` is already registered."""
        if name in self._scanners:
            raise ConfigurationError(f"A scanner named '{name}' is already registered")
        self._scanners[name] = scanner

    def get(self, name: str) -> Scanner:
        """Return the scanner registered under `name`. Raises ConfigurationError if none is registered."""
        try:
            return self._scanners[name]
        except KeyError:
            raise ConfigurationError(f"No scanner named '{name}' is registered") from None

    def list_scanners(self) -> Tuple[ScannerRegistration, ...]:
        """Return every registration, sorted by name for ordering that doesn't depend on registration order."""
        return tuple(ScannerRegistration(name=name, scanner=self._scanners[name]) for name in sorted(self._scanners))
