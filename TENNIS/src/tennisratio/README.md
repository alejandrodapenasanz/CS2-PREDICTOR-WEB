# TennisRatio sidecar

This package acquires a public, daily supplement to the immutable Sackmann
base. It never replaces Sackmann and never writes the operational
`predictions`, `observations` or `settlements` tables.

Authorized network surface: GET requests only to `/robots.txt`,
`/atp-matches.html`, `/wta-matches.html`, `/sitemap-players.xml` and public
`/players/*.html` profiles on `https://www.tennisratio.com`. `/api/` is not an
authorized route. The production transport is the pinned, static
`ScraplingHttpSession` with its fixed Chrome TLS identity and coherent stealth
headers; proxies, browser automation and redirects remain disabled, and tests
can inject an offline requests-like session. The shared responsible client
validates and caches `robots.txt` before any later request, serializes per host
and enforces the one-second minimum. Query-only wildcard rules such as `/*?q=`
are matched as wildcards and cannot be widened accidentally to block every path
beginning with `/`.

The lateral schema keeps a dedicated `source_match_key` index so bilateral
conflict validation remains usable while the first player inventory is loaded.
It is created additively when an existing sidecar is opened and never touches
the operational database or its sacred tables.

TennisRatio describes its dataset as **CC BY-NC 4.0** in the homepage JSON-LD.
Every last-good manifest preserves the attribution
`TennisRatio.com, CC BY-NC 4.0` and the license URL
`https://creativecommons.org/licenses/by-nc/4.0/`.

Raw bytes are content-addressed, gzip-compressed and immutable. Each logical
resource retains at most two blobs; the active last-good blob is protected
from failed attempts. Immutable batch manifests also retain at most two, while
`last_good.json` is atomically reconciled only from a SQLite-published batch.
For profiles, a malformed fetch is recorded as `attempted_compressed_path` but
never replaces or evicts the latest successfully parsed `compressed_path`.

The lateral SQLite database is append-only. Source observations include URL,
SHA-256 and `first_seen_at_utc`. Identity mapping is exact and unique by gender,
normalized full name and DOB when the profile exposes one. Ambiguous/unmapped
profiles and bilateral result conflicts are quarantined. Odds remain only in
raw/audit payloads and are not returned by the mapped result API.
Known surface names in profile history are canonicalized case-insensitively;
an absent surface stays empty and therefore becomes `None` for surface Elo.
Non-finite audit-only odds become null while their original bytes remain in the
immutable raw snapshot.

For a prediction date `D`, use `load_mapped_rankings(D)`,
`load_mapped_results(D, ...)`, `load_identity_resolutions(as_of_date=D)` and
`load_agenda_as_of(D, match_date=D)`. They require effective, observed and
published civil dates strictly before `D`. Calling
`load_identity_resolutions(as_of_date=None)` is reserved for current
presentation/active-agenda mapping and must never feed historical serving or a
feature. `load_active_agenda()` is likewise current presentation state.
It reads exclusively from the newest globally published refresh batch, so a
match removed by that batch cannot reappear from an older schedule.

After updating the local Sackmann masters, run
`python scripts/update_tennisratio.py --remap-only` or invoke the normal refresh
again. If the HTTP batch already ran that UTC day, the second invocation makes
zero GETs and reevaluates stored profile facts locally. Changed resolutions are
published append-only through `identity_remap_batches`; unchanged decisions do
not add rows. Their availability remains strictly gated by the remap event and
publication dates.
