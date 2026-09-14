# uv Package Manager

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager.
This project uses `pyproject.toml` for dependencies and commits `uv.lock` for
reproducible installations.

## Installation

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatively, install uv with `pip install uv`. Restart the terminal if the
`uv` command is not found after installation, then verify it with `uv --version`.

## Project Setup

From the repository root, install the locked dependencies:

```bash
uv sync --frozen
```

`uv sync` creates `.venv` automatically. This project supports Python 3.11 or
newer. To use a specific Python version, run:

```bash
uv sync --python 3.12 --frozen
```

To install an optional integration, include its extra:

```bash
uv sync --extra hf-asr --frozen
uv sync --extra ibm-stt --frozen
```

## Activate `.venv`

Activation is optional: `uv run <command>` automatically uses the project's
`.venv`. If you want to run `python` or project commands directly, activate the
environment for your shell.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow scripts for the current
terminal session and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```batch
.venv\Scripts\activate.bat
```

### Windows Git Bash

```bash
source .venv/Scripts/activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

To leave an activated environment on any platform:

```bash
deactivate
```

## Common Commands

```bash
uv run python run.py               # Run the standard project runner
uv run balsam-eval --help          # Run the project CLI
uv add package-name                # Add and lock a dependency
uv add "package>=1.0.0,<2.0.0"    # Add a version constraint
uv add --dev package-name          # Add a development dependency
uv remove package-name             # Remove a dependency
uv sync --frozen                   # Install exactly from uv.lock
uv sync --upgrade                  # Upgrade all allowed dependencies
uv lock                            # Update uv.lock without installing
```

## Tips

- Run commands from the repository root so uv can find `pyproject.toml`.
- Commit `uv.lock` whenever dependency changes intentionally update it.
- Use `uv run` in scripts and CI so manual activation is unnecessary.
- Use `uv sync --frozen` in CI to fail if `pyproject.toml` and `uv.lock` disagree.
- Do not commit `.venv`; it is machine-specific and can be recreated by uv.
