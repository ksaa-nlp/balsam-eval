# UV Package Manager

Fast Python package manager (10-100x faster than pip) that works with your existing `pyproject.toml`.

## Installation

```bash
pip install uv
```

## Setup

```bash
# Create and activate virtual environment
uv venv
uv venv --python 3.12.8 # if need spicify python version

source .venv/bin/activate  # Linux/macOS

# Install dependencies
uv sync
```

## Common Commands

```bash
uv add package-name              # Add dependency
uv remove package-name           # Remove dependency  
uv add "package>=1.0.0,<2.0.0"   # Add with version constraint
uv pip install -e .               # install without lock file
uv sync --upgrade                 # Update all dependencies
uv sync --frozen
uv run python script.py           # Run command in locked environment
uv lock                          # Generate/update lock file
```


## Tips

- **Commit `uv.lock`** to version control for reproducible installs
- **Use `uv run`** to execute commands without activating venv  
- **Pin versions** in `pyproject.toml` for production stability
