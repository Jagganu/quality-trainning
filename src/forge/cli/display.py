"""Rich-based display utilities for the ForgeGravity CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

if TYPE_CHECKING:
    from forge.core.models import DryRunPlan, GateResult

console = Console()


def show_header() -> None:
    """Print the ForgeGravity banner."""
    console.print(
        Panel(
            "[bold cyan]ForgeGravity[/bold cyan]\n"
            "[dim]Build high-quality AI training datasets[/dim]",
            border_style="cyan",
        )
    )


def show_dry_run(plan: DryRunPlan) -> None:
    """Display a dry-run plan as a Rich table."""
    table = Table(
        title="[bold]Dry Run Plan[/bold]",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Topic", plan.topic)
    table.add_row("Template", plan.template)
    table.add_row("Format", plan.format)
    table.add_row("Estimated Pages", str(plan.estimated_pages))
    table.add_row("Estimated Documents", str(plan.estimated_documents))
    table.add_row("Estimated Samples", str(plan.estimated_samples))
    table.add_row("Estimated Tokens", f"{plan.estimated_tokens:,}")

    cost = plan.estimated_cost
    if cost < 1.0:
        cost_str = f"[green]${cost:.4f}[/green]"
    elif cost < 5.0:
        cost_str = f"[yellow]${cost:.4f}[/yellow]"
    else:
        cost_str = f"[red]${cost:.4f}[/red]"
    table.add_row("Estimated Cost", cost_str)
    table.add_row("Estimated Runtime", plan.estimated_runtime)

    console.print(table)

    if plan.warnings:
        console.print("\n[yellow bold]Warnings:[/yellow bold]")
        for warning in plan.warnings:
            console.print(f"  [yellow]- {warning}[/yellow]")


def show_result(result: object) -> None:
    """Display pipeline completion summary."""
    run = getattr(result, "run_metadata", None)
    metrics = getattr(result, "metrics", None)
    samples = getattr(result, "samples", [])
    output_dir = getattr(result, "output_dir", "")

    table = Table(title="[bold]Pipeline Complete[/bold]", show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Status", f"[green]{run.status.value}[/green]" if run else "unknown")
    table.add_row("Samples Generated", str(len(samples)))

    if metrics:
        vr = metrics.verification_report
        if vr and vr.total_verified > 0:
            rate = vr.pass_rate * 100
            color = "green" if rate >= 80 else "yellow" if rate >= 60 else "red"
            table.add_row("Pass Rate", f"[{color}]{rate:.0f}%[/{color}]")

        cr = metrics.cost_report
        if cr:
            table.add_row("Total Cost", f"${cr.total_cost:.4f}")

    table.add_row("Output", output_dir or "N/A")
    console.print(table)


def show_gate_failure(failed_gates: list[GateResult]) -> None:
    """Display quality gate failures."""
    console.print("\n[red bold]Quality Gate Failures[/red bold]")
    table = Table(show_header=True, header_style="bold red")
    table.add_column("Gate", style="white")
    table.add_column("Actual", style="yellow")
    table.add_column("Threshold", style="cyan")
    table.add_column("Message", style="dim")

    for gate in failed_gates:
        table.add_row(
            gate.gate,
            f"{gate.actual_value:.3f}",
            f"{gate.threshold:.3f}",
            gate.message,
        )

    console.print(table)


def make_progress() -> Progress:
    """Create a Rich Progress bar for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )
