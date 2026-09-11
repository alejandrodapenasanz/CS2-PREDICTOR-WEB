# Verified local wheel

This directory contains the one vendored artifact required to resolve the
scraper environment reproducibly:

```text
flasgger-0.9.7.1-py2.py3-none-any.whl
SHA256: a25e4a666b2894ba8833acec2d0827646e675f914f25664d68bc88f65e2e7783
```

The wheel metadata must identify distribution `flasgger`, version `0.9.7.1`.
No other wheel is allowlisted by the component's `.gitignore`.

Verify the checked-in artifact without network access:

```powershell
py -3.13 scripts\verify_flasgger_wheel.py
```

## Verified reproducible build recipe

The source artifact used for the allowlisted wheel is:

```text
flasgger-0.9.7.1.tar.gz
SHA256: ca098e10bfbb12f047acc6299cc70a33851943a746e550d86e65e60d4df245fb
```

The byte-reproducible build was verified with this exact toolchain:

- CPython 3.13.15
- pip 25.2
- setuptools 84.0.0
- wheel 0.47.0
- `SOURCE_DATE_EPOCH=1684430121`

From a fresh temporary directory containing the verified sdist, create an
isolated build environment and assert the interpreter patch version:

```powershell
py -3.13 -m venv .wheel-build
.\.wheel-build\Scripts\python.exe -c "import platform; assert platform.python_version() == '3.13.15'"
.\.wheel-build\Scripts\python.exe -m pip install pip==25.2 setuptools==84.0.0 wheel==0.47.0
if ((Get-FileHash .\flasgger-0.9.7.1.tar.gz -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ca098e10bfbb12f047acc6299cc70a33851943a746e550d86e65e60d4df245fb') { throw 'flasgger sdist SHA-256 mismatch' }
$env:SOURCE_DATE_EPOCH = '1684430121'
```

Run the same PEP 517 build twice into separate empty output directories:

```powershell
.\.wheel-build\Scripts\python.exe -m pip wheel --use-pep517 --no-build-isolation --no-deps --no-cache-dir --wheel-dir .\wheel-build-1 .\flasgger-0.9.7.1.tar.gz
.\.wheel-build\Scripts\python.exe -m pip wheel --use-pep517 --no-build-isolation --no-deps --no-cache-dir --wheel-dir .\wheel-build-2 .\flasgger-0.9.7.1.tar.gz
Get-FileHash .\wheel-build-1\flasgger-0.9.7.1-py2.py3-none-any.whl -Algorithm SHA256
Get-FileHash .\wheel-build-2\flasgger-0.9.7.1-py2.py3-none-any.whl -Algorithm SHA256
```

This recipe was executed in multiple clean builds. Every output was
byte-identical and produced:

```text
SHA256: a25e4a666b2894ba8833acec2d0827646e675f914f25664d68bc88f65e2e7783
```

Dependency installers must use `requirements.lock.txt`, `--require-hashes`,
`--only-binary=:all:`, `--no-deps`, and `--find-links wheels`. The input
`requirements.txt` is maintained for lock generation only.

Do not replace or rebuild this binary silently. A deliberate replacement
requires reviewing its source and wheel metadata, repeating the clean-build
comparison, updating the expected digest in the verifier and this document,
regenerating the lock under CPython 3.13, and running the offline scraper tests.
