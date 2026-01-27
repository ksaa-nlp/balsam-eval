# UV Quick Reference Cheat Sheet

## 🚀 Setup (One Time)
```bash
uv venv                          # Create virtual environment
source .venv/bin/activate        # Activate (Linux/macOS)
.venv\Scripts\activate           # Activate (Windows)
uv lock                          # Generate lock file
uv sync                          # Install dependencies
```

## 📦 Daily Commands
```bash
# Pull changes and sync
git pull
uv sync --frozen

# Activate environment
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

## ➕ Adding Dependency
```bash
# 1. Edit pyproject.toml, add: "package==1.0.0"
# 2. Update lock file
uv lock
# 3. Install
uv sync
# 4. Commit
git add pyproject.toml uv.lock
git commit -m "Add package"
```

## 🔄 Updating Dependencies
```bash
# Update all
uv lock --upgrade
uv sync

# Update specific package
uv lock --upgrade-package transformers
uv sync
```

## 🐳 Docker
```bash
docker build -t app .
docker run -e API_KEY=key -e SERVER_TOKEN=token app
```

## 🔍 Troubleshooting
```bash
# Lock file out of sync
uv lock

# Clean reinstall
uv cache clean
rm -rf .venv
uv venv
source .venv/bin/activate
uv sync --frozen

# Check what's installed
uv pip list
```

## ⚡ Makefile Shortcuts
```bash
make install       # Full setup
make lock          # Update lock file
make sync          # Install from lock
make test          # Run tests
make update        # Update all deps
make clean         # Clean everything
```

## 🎯 Remember
- ✅ Always commit `uv.lock`
- ✅ Use `uv sync --frozen` in Docker/CI
- ✅ Run `uv lock` after changing pyproject.toml
- ✅ Keep virtual environment activated
- ❌ Never edit uv.lock manually
- ❌ Never add uv.lock to .gitignore

## 📖 Full Documentation
See `UV-GUIDE.md` for complete guide