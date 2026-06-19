# ForgeGravity

Build high-quality AI training datasets from raw knowledge.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

ForgeGravity is an open-source toolkit for creating synthetic training datasets. It runs an async, multi-stage pipeline that collects domain knowledge, cleans source material, generates diverse samples with LLMs, verifies output quality, and exports datasets with cost and lineage tracking.

## Why ForgeGravity?

- End-to-end dataset pipeline: collect, clean, analyze, generate, verify, refine, benchmark, export.
- Built-in templates for cybersecurity, reasoning, and coding datasets.
- Multiple dataset formats: reasoning, instruction, agent, coding, and chat.
- Quality gates for duplicate rate, diversity, verification score, and budget limits.
- Provenance tracking through `SampleLineage` and structured pipeline metadata.

## Architecture

```text
Collect -> Clean -> Analyze -> Generate -> Verify -> Refine
                                      \-> Benchmark -> Export
```

Core concepts:

| Concept | Description |
|---------|-------------|
| `PipelineContext` | Shared state passed through every stage. |
| `Stage` | Abstract base class for pipeline steps. |
| `CostBudget` | Tracks spend and enforces hard budget limits. |
| `QualityGate` | Checks final dataset quality before completion. |
| `HookManager` | Async lifecycle hooks for observability and plugins. |
| `SampleLineage` | Tracks source documents, models, run IDs, and stage history. |

## Quick Start

```bash
git clone https://github.com/<your-org>/forgegravity.git
cd forgegravity
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On macOS or Linux, use `python3` instead of `py` and activate the virtual environment normally.

Preview a run without making API calls:

```bash
forge build cybersecurity --dry-run
```

Generate a dataset:

```bash
forge build cybersecurity --format reasoning --max-samples 50
```

Validate an existing JSONL dataset:

```bash
forge validate ./output/cybersecurity_dataset.jsonl --format reasoning
```

List built-in templates:

```bash
forge list-templates
```

## CLI

| Command | Description |
|---------|-------------|
| `forge build <topic>` | Run the full pipeline for a topic or template name. |
| `forge build <topic> --dry-run` | Show estimated cost, pages, and samples without executing. |
| `forge build <topic> --max-cost 2.0` | Set a hard budget cap in USD. |
| `forge build <topic> --model gpt-4o-mini` | Override the default LLM. |
| `forge list-templates` | Show built-in templates and estimated output. |
| `forge validate <path> --format <fmt>` | Validate an existing JSONL dataset. |

Common flags:

```text
--config, -c PATH    Path to forge.toml config file
--output, -o PATH    Output directory (default: ./output)
--format, -f TEXT    Dataset format: reasoning, instruction, agent, coding, chat
--model,  -m TEXT    LLM model identifier supported by LiteLLM
--max-samples INT    Cap total generated samples
--max-cost FLOAT     Hard budget limit in USD
--verbose, -v        Enable debug logging
```

## Configuration

ForgeGravity loads settings from highest to lowest priority:

1. Environment variables prefixed with `FORGE_`
2. `.env`
3. `forge.toml`
4. Built-in defaults

Example:

```toml
[general]
default_model = "gpt-4o-mini"
output_dir = "./output"
log_level = "INFO"

[collect]
max_pages = 50
max_documents = 20
requests_per_second = 2.0

[generate]
default_format = "reasoning"
max_samples = 100
batch_size = 10
temperature = 0.7

[verify]
enabled = true
critic_model = "gpt-4o-mini"
min_score = 0.7

[budget]
max_cost_usd = 5.0
warn_at_percent = 80.0

[quality_gates]
max_duplicate_rate = 0.10
min_diversity_score = 0.5
min_verification_score = 0.7
```

## Templates

| Template | Format | Estimated Samples | Description |
|----------|--------|-------------------|-------------|
| `cybersecurity` | reasoning | 200 | Vulnerabilities, CTF, secure coding, and incident response. |
| `reasoning` | reasoning | 100 | Logic, probability, critical thinking, and scientific method. |
| `coding` | coding | 150 | Algorithms, debugging, design patterns, and code review. |

## Dataset Formats

| Format | Required Fields | Use Case |
|--------|-----------------|----------|
| `reasoning` | `question`, `analysis`, `answer`, `metadata` | Reasoning fine-tuning records. |
| `instruction` | `messages` | ChatML-style instruction data. |
| `agent` | `observation`, `thought`, `action`, `result` | ReAct-style agent traces. |
| `coding` | `issue`, `investigation`, `patch`, `verification` | Software engineering tasks. |
| `chat` | `conversations` | Multi-turn dialogue datasets. |

## Development

```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src/forge
```

## Project Layout

```text
src/forge/
  cli/             Typer CLI and Rich display helpers
  core/            Pipeline engine, settings, models, hooks, gates, budget
  stages/          Collect, clean, analyze, generate, verify, refine, benchmark, export
  verification/    Critics, scorers, validators, and consensus
  datasets/        Schemas, validators, loaders, and exporters
  trajectories/    Agent trajectory recording and formatting
  templates/       Domain-specific dataset blueprints
  providers/       LLM provider abstraction
  metrics/         Counters, timers, and reports
  storage/         Filesystem and SQLite backends
  registry/        Plugin registries
  utils/           Logging, cost, and hashing helpers
tests/             Unit tests for core, datasets, and verification
```

## License

MIT. See [LICENSE](LICENSE) for details.
