# Contributing to M.E.M.I.R.

Thank you for your interest in contributing! This guide will help you set up your development environment and follow best practices for code quality and collaboration.

## Development Setup

1. **Clone the repository:**
   ```sh
   git clone <repo_url>
   cd Memir
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   pip install black isort flake8
   ```

## Code Style & Quality

- **Formatting:** Run `black .` to auto-format code.
- **Import Sorting:** Run `isort .` to sort imports.
- **Linting:** Run `flake8 .` to check for style issues.
- **Line Length:** 88 characters (Black default).
- **Config:** See `pyproject.toml` and `.flake8` for tool settings.

## Running Tests

- Use `pytest` or your preferred test runner.
- Example: `python test_chromadb.py`

## Submitting Changes

1. Format, sort, and lint your code before committing.
2. Describe your changes clearly in commit messages.
3. Update `CHANGELOG.md` with notable changes.
4. Submit a pull request for review.

---

*May your code be as wise and enduring as Mimir himself.*
