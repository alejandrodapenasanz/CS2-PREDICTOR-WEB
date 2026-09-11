# HLTV scraper component

This directory is the isolated Python environment used by the CS2 daily
pipeline and the optional Flask API. Its dependency target is **CPython 3.13**.
Repository moves are supported: Python modules are launched through the active
interpreter, never through a generated `scrapy.exe` that embeds an old path.

## Dependency contract

The two dependency files have different responsibilities:

- `requirements.txt` lists direct dependencies with reviewed compatible
  ranges. It is an input for maintainers; production does not install it.
- `requirements.lock.txt` fixes the complete transitive graph and hashes. It
  is the only installation input for the pipeline, Docker, and Make.

The direct contract includes Flask/flasgger, Scrapy, requests/parsel,
cloudscraper, scrapy-impersonate/curl-cffi, nodriver,
`scrapling[fetchers]`, and the offline test tools. Selenium and
scrapy-selenium are not dependencies of this component.

Flasgger 0.9.7.1 is supplied as one verified pure-Python wheel under
`wheels/`. Its allowlisted SHA-256 and offline verification procedure are
documented in [wheels/README.md](wheels/README.md).

## Recreate and install

Do not move or copy an existing virtual environment. Recreate it at the final
repository path with Python 3.13:

```powershell
Set-Location C:\dev\CS2-Predictor\CS2\SCRAPER\hltv-scraper-api
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe scripts\verify_flasgger_wheel.py
.\.venv\Scripts\python.exe -m pip install --no-deps --only-binary=:all: --require-hashes --find-links wheels -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m scrapy version
```

The root CS2 launcher provisions this same environment. A changed lock,
interpreter contract, or local-wheel digest invalidates the environment and
must cause recreation instead of an unpinned repair.

For POSIX/Make, point `PYTHON` at a Python 3.13 environment:

```bash
python3.13 -m venv .venv
make PYTHON=.venv/bin/python install
make PYTHON=.venv/bin/python test-unit
```

On Windows with GNU Make, use
`PYTHON=.venv/Scripts/python.exe`.

## Lock maintenance

Only maintainers regenerate the lock, under CPython 3.13 and from this
directory. `pip-tools==7.5.2` is a pinned maintainer tool, not a scraper
runtime dependency. Verify the wheel first, then run this exact command:

```powershell
py -3.13 scripts\verify_flasgger_wheel.py
py -3.13 -m pip install pip-tools==7.5.2
py -3.13 -m piptools compile --resolver=backtracking --generate-hashes --allow-unsafe --strip-extras --no-emit-index-url --find-links ./wheels --no-emit-find-links --pip-args '--only-binary=:all:' --output-file requirements.lock.txt requirements.txt
```

Review the complete diff and install the new lock into a clean environment
before committing it. The browser assets installed by Scrapling are outside
pip's lock and must be provisioned by the root launcher.

## Launching

Run Scrapy through the environment's interpreter:

```powershell
.\.venv\Scripts\python.exe -m scrapy version
.\.venv\Scripts\python.exe -m scrapy crawl hltv_upcoming_matches
```

Run the optional API from the component directory:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Offline tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dependency_contract.py tests\test_routes.py -q
```

The dependency-contract tests do not contact HLTV. Tests marked
`integration` or `slow` may use the external site and are not part of the
offline verification command.

## Local quality gates

The component owns a Python 3.13 gate configuration in `pyproject.toml`.
After installing the lock, run the same deterministic checks used by CI:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe scripts\check_import_coverage.py
.\.venv\Scripts\python.exe -m pytest -m "not slow and not integration" tests
.\.venv\Scripts\python.exe scripts\smoke_boot.py
```

On POSIX, `make PYTHON=.venv/bin/python gates` runs the same sequence. The
smoke is offline and fixed to seed 42: it creates the real Flask app, loads
the production Scrapy project and all registered spiders, and boots both
collection CLIs through `--help`.

`check_import_coverage.py` parses every Python syntax tree in production and
tests, then fails if a third-party import lacks a direct declaration in
`requirements.txt`. `test_import_coverage_detects_removed_dependency` proves
the failure path by removing `requests` from a temporary copy of the direct
requirements and then verifies the restored, real file still passes.

### Incremental debt kept visible

- Mypy currently covers the Flask app, routes, collector scripts, dependency
  tooling, and the top-level scraper manager/cache modules. The legacy inner
  `hltv_scraper/hltv_scraper` spiders and parsers still need an annotation and
  abstract-parser-signature cleanup before they can join the typed surface.
- Ruff intentionally ignores `F401` for now because some Scrapy imports are
  registration hooks. Replace those broad suppressions with per-file ignores
  before enabling the rule globally.
- Live HLTV integration tests remain opt-in because network availability and
  anti-bot responses are nondeterministic; the default gate excludes both
  `integration` and `slow` markers.
