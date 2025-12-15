# GitHub Actions Workflows

This directory contains CI/CD workflows for the Low-Rank Toolbox project.

## Workflows

### 🧪 Tests (`tests.yml`)
- **Trigger:** Push to main/develop, pull requests
- **Matrix:** Python 3.10, 3.11, 3.12 × Ubuntu, macOS, Windows
- **Actions:**
  - Runs full test suite (1000+ tests) with pytest
  - Generates code coverage reports
  - Uploads coverage to Codecov
  - Uploads test results as artifacts
  - Ensures cross-platform compatibility

### 📚 Documentation (`documentation.yml`)
- **Trigger:** Push to main/develop, pull requests
- **Actions:**
  - Builds Sphinx HTML documentation
  - Checks for broken links
  - Uploads docs as artifacts
  - **Auto-deploys to GitHub Pages** (main branch only)

### 🎨 Code Quality (`code-quality.yml`)
- **Trigger:** Push to main/develop, pull requests
- **Actions:**
  - Checks code formatting (black)
  - Verifies import sorting (isort)
  - Lints code (flake8)
  - Type checking (mypy)
  - All checks are informational (continue-on-error)

## Viewing Results

### Test Results
Check the **Actions** tab on GitHub for test results across all platforms.

### Coverage Reports
- Coverage reports are automatically uploaded to Codecov
- View detailed coverage metrics and trends at your Codecov dashboard

### Documentation
- **Preview:** Download the `documentation-html` artifact from any workflow run
- **Live Docs:** Automatically published to GitHub Pages from main branch
  - URL: `https://BenjaminCarrel.github.io/low-rank-toolbox/`

### Code Quality
Review code quality checks in the workflow summary.

## Local Testing

Before pushing, run these locally:

```bash
# Activate the conda environment
conda activate low-rank-dev

# Run tests with coverage
pytest -v --cov=src/lowrank --cov-report=term

# Check code formatting
black --check src/ tests/

# Check import sorting
isort --check-only src/ tests/

# Lint code
flake8 src/

# Type check
mypy src/lowrank --ignore-missing-imports

# Build documentation
cd docs
make html
```

## Setup Notes

All workflows use:
- **conda-incubator/setup-miniconda@v3** for Python environment management
- **environment.yml** for consistent dependency installation
- Python versions 3.10, 3.11, and 3.12 (matching pyproject.toml requirement)

pytest -v

# Build documentation
cd docs && make html

# Format code (optional)
black src/ tests/
isort src/ tests/
flake8 src/
```

## Enabling GitHub Pages

To enable automatic documentation deployment:

1. Go to repository **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: **gh-pages** / **root**
4. Save

The `documentation.yml` workflow will automatically publish docs to this site on pushes to main.

## Workflow Badges

Add these to your README.md:

```markdown
![Tests](https://github.com/BenjaminCarrel/low-rank-toolbox/actions/workflows/tests.yml/badge.svg)
![Documentation](https://github.com/BenjaminCarrel/low-rank-toolbox/actions/workflows/documentation.yml/badge.svg)
![Code Quality](https://github.com/BenjaminCarrel/low-rank-toolbox/actions/workflows/code-quality.yml/badge.svg)
```
