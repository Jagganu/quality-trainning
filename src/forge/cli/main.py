"""ForgeGravity CLI - Typer application with build, validate, and list-templates commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="forge",
    help="ForgeGravity - Build high-quality AI training datasets",
    no_args_is_help=True,
)


@app.command()
def build(
    topic: str = typer.Argument(..., help="Topic or template name to build dataset for"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to forge.toml"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    format: str = typer.Option("reasoning", "--format", "-f", help="Dataset format"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Max samples to generate"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Max USD to spend"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without executing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Build a training dataset for the given topic."""
    from forge.cli.display import console, show_dry_run, show_gate_failure, show_header, show_result
    from forge.core.budget import BudgetExceededError
    from forge.core.config import load_settings
    from forge.core.gates import QualityGateFailedError
    from forge.core.pipeline import Pipeline

    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    settings = load_settings(config)
    if output:
        settings.output_dir = str(output)
    if model:
        settings.default_model = model
    if max_samples:
        settings.generate.max_samples = max_samples
    if max_cost is not None:
        settings.budget.max_cost_usd = max_cost
    settings.generate.default_format = format

    pipeline = Pipeline(settings)

    show_header()

    if dry_run:
        plan = asyncio.run(pipeline.dry_run(topic))
        show_dry_run(plan)
        return

    console.print(f"\n[bold cyan]Building dataset for:[/bold cyan] {topic}")
    console.print(f"[dim]Model: {settings.default_model} | Format: {format}[/dim]\n")

    try:
        result = asyncio.run(pipeline.run(topic))
        show_result(result)
    except QualityGateFailedError as exc:
        show_gate_failure(exc.failed_gates)
        raise typer.Exit(code=1)
    except BudgetExceededError as exc:
        console.print(f"\n[red bold]Budget exceeded:[/red bold] {exc}")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(code=130)


@app.command("list-templates")
def list_templates() -> None:
    """List available built-in templates."""
    from rich.table import Table

    from forge.cli.display import console
    from forge.templates import TEMPLATES

    table = Table(
        title="[bold]Available Templates[/bold]",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Format", style="green")
    table.add_column("Est. Samples", style="yellow", justify="right")

    for name, cls in TEMPLATES.items():
        tmpl = cls()
        table.add_row(name, tmpl.description, tmpl.default_format, str(tmpl.estimated_sample_count()))

    console.print(table)


@app.command()
def validate(
    path: Path = typer.Argument(..., help="Path to JSONL dataset to validate"),
    format: str = typer.Option("reasoning", "--format", "-f", help="Expected dataset format"),
) -> None:
    """Validate an existing JSONL dataset."""
    import json

    from forge.cli.display import console
    from forge.datasets.validators import DatasetValidator

    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(code=1)

    records: list[dict] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    console.print(f"[yellow]  Line {line_num}: skipped — {exc}[/yellow]")
                    skipped += 1

    if skipped:
        console.print(f"[yellow]Skipped {skipped} malformed line(s)[/yellow]")

    if not records:
        console.print("[red]No valid records found[/red]")
        raise typer.Exit(code=1)

    validator = DatasetValidator()
    report = validator.full_validation(records, format)

    console.print(f"\n[bold]Validation Report for {path.name}[/bold]")
    console.print(f"  Total records:  {report.total_records}")
    console.print(f"  Valid records:  {report.valid_records}")

    rate = report.pass_rate * 100
    color = "green" if rate >= 90 else "yellow" if rate >= 70 else "red"
    console.print(f"  Pass rate:      [{color}]{rate:.0f}%[/{color}]")

    if report.errors:
        console.print(f"\n[red]Errors ({len(report.errors)}):[/red]")
        for err in report.errors[:20]:
            console.print(f"  [red]- {err}[/red]")
        if len(report.errors) > 20:
            console.print(f"  [dim]... and {len(report.errors) - 20} more[/dim]")

    if report.warnings:
        console.print(f"\n[yellow]Warnings ({len(report.warnings)}):[/yellow]")
        for warn in report.warnings[:10]:
            console.print(f"  [yellow]- {warn}[/yellow]")


if __name__ == "__main__":
    app()
