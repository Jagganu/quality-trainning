"""Report generation — Rich console, JSON, and HTML output."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forge.core.models import DatasetMetrics
from forge.utils.cost import format_cost


class ReportGenerator:
    """Produces human-readable reports from :class:`DatasetMetrics`."""

    def console_summary(self, metrics: DatasetMetrics, console: Console | None = None) -> None:
        """Print a Rich summary table to the console."""
        con = console or Console()

        table = Table(title="Dataset Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total samples", str(metrics.total_samples))
        table.add_row("Verified samples", str(metrics.verified_samples))
        table.add_row("Rejected samples", str(metrics.rejected_samples))
        table.add_row("Diversity score", f"{metrics.diversity_score.overall:.2f}")
        table.add_row("Duplicates (exact)", str(metrics.deduplication_report.exact_duplicates))
        table.add_row("Duplicates (near)", str(metrics.deduplication_report.near_duplicates))
        table.add_row("Total cost", format_cost(metrics.cost_report.total_cost))
        table.add_row("Tokens in", f"{metrics.cost_report.total_tokens_in:,}")
        table.add_row("Tokens out", f"{metrics.cost_report.total_tokens_out:,}")

        if metrics.stage_durations:
            table.add_section()
            for stage, dur in metrics.stage_durations.items():
                table.add_row(f"  ⏱ {stage}", f"{dur:.1f}s")

        con.print(table)

    def json_report(self, metrics: DatasetMetrics, path: str) -> None:
        """Write metrics to a JSON file."""
        Path(path).write_text(
            metrics.model_dump_json(indent=2), encoding="utf-8"
        )

    def html_report(self, metrics: DatasetMetrics, path: str) -> None:
        """Write a simple HTML report."""
        data = metrics.model_dump()
        rows = ""
        for key, val in data.items():
            if isinstance(val, dict):
                val = json.dumps(val, indent=2)
            rows += f"<tr><td><b>{key}</b></td><td><pre>{val}</pre></td></tr>\n"

        html = f"""<!DOCTYPE html>
<html><head><title>ForgeGravity Report</title>
<style>body{{font-family:system-ui;margin:2em}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#1a1a2e;color:#fff}}</style>
</head><body><h1>ForgeGravity Dataset Report</h1>
<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table></body></html>"""
        Path(path).write_text(html, encoding="utf-8")
