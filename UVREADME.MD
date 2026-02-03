# UV Usage Guide for balsam-lm-evaluation

## Installation

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

## Common Commands

### Project Setup

**Create a virtual environment:**
```bash
uv venv
```

**Activate the virtual environment:**
```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**Install project dependencies:**
```bash
uv pip install -e .
```

**Install dependencies without installing the project:**
```bash
uv pip install -r pyproject.toml
```

### Dependency Management

**Add a new dependency:**
```bash
# Edit pyproject.toml manually and add to dependencies list
# Then sync:
uv pip install -e .
```

**Install a specific package:**
```bash
uv pip install package-name
```

**Install with version constraints:**
```bash
uv pip install "package-name>=1.0.0,<2.0.0"
```

**Upgrade a package:**
```bash
uv pip install --upgrade package-name
```

**Uninstall a package:**
```bash
uv pip uninstall package-name
```

**List installed packages:**
```bash
uv pip list
```

**Freeze dependencies:**
```bash
uv pip freeze > requirements.txt
```

### Lock File

**Generate a lock file:**
```bash
uv pip compile pyproject.toml -o requirements.lock
```

**Install from lock file:**
```bash
uv pip sync requirements.lock
```

## Development Workflow

### Initial Setup
```bash
# Clone your repository
git clone <your-repo-url>
cd balsam-lm-evaluation

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install project in editable mode
uv pip install -e .
```

### Running Tests
```bash
# Ensure virtual environment is activated
pytest
```

### Running Your Application
```bash
python run.py
```

## Docker Build

**Build the image:**
```bash
docker build -t balsam-lm-evaluation .
```

**Run the container:**
```bash
docker run -e API_KEY=your_key -e SERVER_TOKEN=your_token balsam-lm-evaluation
```

## Cloud Build

Your existing Cloud Build configurations will work with the new Dockerfile. Just trigger a build:

```bash
# Production
gcloud builds submit --config=cloudbuild.yaml

# Development
gcloud builds submit --config=cloudbuild-dev.yaml
```

## Key Differences from Poetry

| Task | Poetry | uv |
|------|--------|-----|
| Install deps | `poetry install` | `uv pip install -e .` |
| Add dependency | `poetry add package` | Edit `pyproject.toml` + `uv pip install -e .` |
| Remove dependency | `poetry remove package` | Edit `pyproject.toml` + `uv pip install -e .` |
| Update deps | `poetry update` | `uv pip install --upgrade -e .` |
| Run command | `poetry run python` | Just `python` (with venv activated) |
| Virtual env | `poetry shell` | `source .venv/bin/activate` |

## Performance Benefits

uv is significantly faster than pip and Poetry:
- **10-100x faster** package installation
- **Parallel downloads** and installations
- **Better dependency resolution**
- **Written in Rust** for maximum performance

## Tips

1. **Always activate your virtual environment** before running commands
2. **Use `uv pip` instead of `pip`** for faster operations
3. **Keep your `pyproject.toml` updated** manually when adding/removing dependencies
4. **Commit `uv.lock`** to version control for reproducible builds (optional)
5. **Use `.venv`** as your virtual environment name (uv's default)

## Troubleshooting

**Virtual environment not found:**
```bash
uv venv  # Create it again
```

**Dependencies not installing:**
```bash
# Clear cache and reinstall
uv cache clean
uv pip install -e . --no-cache
```

**Wrong Python version:**
```bash
# Create venv with specific Python version
uv venv --python 3.10
```