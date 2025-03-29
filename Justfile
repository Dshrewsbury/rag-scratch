set dotenv-load

# use PowerShell instead of sh:
#set shell := ["powershell.exe", "-c"]

_list:
  @just --list --unsorted

scaffold:
  @echo "Scaffolding infrastructure..."
  @docker compose up --build

# Remove compose containers from the background
teardown:
  @echo "Tearing down infrastructure..."
  @docker compose down

# Lint and format with **Ruff**
tidy:
  @echo "Linting..."
  @uv run ruff check --fix

  @echo "\nFormatting..."
  @uv run ruff format

type-check:
  @echo "Type checking..."
  @uv run mypy app/
  @uv run mypy rag/
  @uv run mypy ui/

sort:
  @echo "Sorting imports..."
  @uv run isort app/
  @uv run isort rag/
  @uv run isort ui/

# Run (`e2e`, `integration`, `unit`) tests; If none specified, run all tests
test tests="":
  @echo "Testing..."
  @uv run pytest tests/{{tests}} --cov=app/ --cov-report term-missing

ci:
  @just sort
  @echo
  @just tidy
  @echo
  @just type-check
  @echo
  @just test

# `build` or `run` the project container with `args`
container cmd *args:
  @just _container-{{cmd}} {{args}}