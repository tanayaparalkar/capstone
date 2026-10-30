"""
Terminal rendering for `sentinelai providers`: what each provider is, and what leaves the machine.

Data retention has three states and is rendered as three words. A
provider whose retains_data is None has not been verified, and is shown
as "unverified" - never as "no", which would turn an open question into
a privacy claim. Credentials are shown as set/missing only; their values
are never read here.
"""
import os
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..ai.config import DEFAULT_EGRESS, DEFAULT_PROVIDER, PROVIDERS, ProviderSpec


def retention_label(retains_data: Optional[bool]) -> str:
    if retains_data is None:
        return "unverified"
    return "retains data" if retains_data else "not retained"


def location_label(spec: ProviderSpec) -> str:
    return "local" if spec.is_local else "remote"


def credentials_label(spec: ProviderSpec) -> str:
    if not spec.required_env:
        return "none needed"
    return "\n".join(
        f"{name} ({'set' if os.environ.get(name, '').strip() else 'missing'})" for name in spec.required_env
    )


def render_providers(console: Console) -> None:
    table = Table(title="LLM providers", show_header=True, header_style="bold")
    table.add_column("Provider", no_wrap=True)
    table.add_column("Default model", no_wrap=True)
    table.add_column("Runs", no_wrap=True)
    table.add_column("Data retention", no_wrap=True)
    table.add_column("Credentials", overflow="fold")
    for spec in PROVIDERS.values():
        runs = location_label(spec)
        retention = retention_label(spec.retains_data)
        table.add_row(
            spec.name,
            spec.default_model,
            runs if spec.is_local else f"[yellow]{runs}[/yellow]",
            retention if retention != "unverified" else f"[yellow]{retention}[/yellow]",
            credentials_label(spec),
        )
    console.print(table)
    console.print(
        f"Default: {DEFAULT_PROVIDER}, egress {DEFAULT_EGRESS}. A remote provider sends code slices off this "
        "machine and runs only with --ai-provider <name> --egress allow (or SENTINELAI_AI_PROVIDER / "
        "SENTINELAI_EGRESS). \"unverified\" means the provider's data-retention terms have not been checked."
    )
