# CS2 quality backlog

The mandatory local gate is `python scripts/quality_gate.py`. Ruff lint, mypy,
tests, import coverage, entrypoint boot and the deterministic smoke are blocking.

Ruff formatting starts from a deliberately narrow baseline so that introducing
the gate does not mix a repository-wide mechanical rewrite with functional work.
The following legacy areas remain to be brought under `ruff format --check`:

- the remainder of `MODEL/` outside the seven baseline files;
- `BBDD/`;
- `PIPELINE/`;
- the remainder of `TESTS/`.

New or materially edited Python files must be formatted before their baseline is
expanded. The backlog is formatting-only; existing `ruff check` remains blocking
over the complete CS2 source and test surface.
