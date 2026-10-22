# Contributing to gdmbayes

Thank you for your interest in contributing to gdmbayes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/butterman0/gdmbayes.git
cd gdmbayes

# Install in editable mode with dev dependencies (Python ≥ 3.12)
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run the full test suite
pytest src/gdmbayes/tests/ -v

# Run a specific test file
pytest src/gdmbayes/tests/test_gdm.py -v

# Run a single test
pytest src/gdmbayes/tests/test_gdm.py::TestGDM::test_fit -v

# Run with coverage report
pytest src/gdmbayes/tests/ --cov=src/gdmbayes --cov-report=term-missing
```

All tests must pass before submitting a pull request.

## Code Style

gdmbayes uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for issues and auto-fix where possible
ruff check src/ --fix

# Format code
ruff format src/
```

Line length is set to 100 characters. Rule E501 (line too long) is ignored for long
docstring lines and comments. Docstrings follow the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) standard.

## Pre-commit

The same checks run automatically on each commit if you install the hooks:

```bash
pre-commit install
```

## Pull Request Process

`main` is protected: nothing is pushed to it directly, every change — including the
maintainer's — arrives through a pull request whose CI checks pass.

1. Fork the repository (or create a branch if you have write access) from `main`.
2. Write or update tests for any changed behaviour. The test suite lives in
   `src/gdmbayes/tests/`.
3. Ensure all tests pass and `ruff check` / `ruff format --check` report no errors (CI runs both).
4. Add a line to `CHANGELOG.md` under "Unreleased".
5. Open a pull request; the template asks for a short description and a checklist.
   CI must be green before it can be merged (squash merge by default).

## Releases

1. Bump `version` in `pyproject.toml`, `CITATION.cff` and the README BibTeX; move the
   "Unreleased" changelog section to the new version with today's date.
2. Merge via PR, then tag: `git tag -a vX.Y.Z -m "gdmbayes X.Y.Z" && git push origin vX.Y.Z`.
3. Create a GitHub Release from the tag (Zenodo mints the DOI automatically).
4. Build and upload: `python -m build && python -m twine upload dist/*`.

## Project Structure

See `docs/architecture.md` for the package architecture and class hierarchy, and
`docs/design_decisions.md` for key design decisions.

## Reporting Issues

Please open an issue on GitHub with a minimal reproducible example and the full
traceback.
