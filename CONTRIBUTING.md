# Contributing to gdmbayes

Thank you for your interest in contributing to gdmbayes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/<org>/gdmbayes.git
cd gdmbayes

# Install in editable mode with dev dependencies (Python ≥ 3.10)
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run the full test suite
pytest src/gdmbayes/tests/ -v

# Run a specific test file
pytest src/gdmbayes/tests/test_models.py -v

# Run a single test
pytest src/gdmbayes/tests/test_models.py::TestGDM::test_fit -v

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
docstring lines and comments.

## Type Checking

```bash
mypy src/gdmbayes/
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Write or update tests for any changed behaviour. The test suite lives in
   `src/gdmbayes/tests/`.
3. Ensure all tests pass and `ruff check` reports no errors.
4. Summarise your changes in `CHANGELOG.md` under an `Unreleased` section.
5. Open a pull request with a clear description of what the change does and why.

## Project Structure

See `CLAUDE.md` for a detailed description of the package architecture, class hierarchy,
and key design decisions.

## Reporting Issues

Please open an issue on GitHub with a minimal reproducible example and the full
traceback.
