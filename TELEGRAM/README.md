# CS2 Predictor — Telegram publisher

This component publishes the daily CS2 predictions to the read-only Telegram
channel [@cs2DailyPicks](https://t.me/cs2DailyPicks) through
`@CS2PredictorPublisherBot`. It produces exactly two English posts for each
date:

1. **BEST OPPORTUNITIES**, containing every qualified prediction for the date.
2. **TODAY'S OTHER PICKS**, containing all remaining daily predictions.

After a successful publication, it atomically replaces `TELEGRAM/daily_report.txt`
with all newly detected qualified opportunities from the requested date onward.
Exported match IDs are retained in SQLite, so a match never appears in the report
twice even after the text file has been replaced.

The probability shown is a model estimate, not a promise or betting guarantee.

### Selection and confidence (2026-09-05)

Both the first post and the incremental report require upstream eligibility
and `prediction.decision_confidence > 0.65`, compared **before rounding**.
This is the adjusted win probability of `decision_favorite`, the same numeric
field used by the Best Opportunity cards, not the raw model probability.
Following the Telegram request, exactly 65% goes to OTHER PICKS (the web's
existing inclusive >=65% boundary is unchanged). All other validated picks
remain in the second post; a high raw probability never overrides the CS2
eligibility gates. No odds are required by this additional display filter.

Every pick in both posts and the report also shows `Confidence`, copied from
`prediction.estimate_confidence_level`: HIGH, MEDIUM or LOW. It describes
uncertainty in the estimate, not the chance of winning; a >65% pick can have
LOW confidence. Missing/null levels display NOT AVAILABLE; unknown values
fail validation. No confidence is invented from probability or reliability.
Formatting changes never resend already confirmed publication slots. If the
same date was already published with different text, the existing payload-conflict
guard stops the attempt; do not delete/reset its SQLite rows to resend it.

## Security action required before the first real post

The original Bot API token was pasted into a conversation and must therefore be
treated as compromised. Do **not** put that token in `.env` and do not test it.

1. Open the verified [`@BotFather`](https://t.me/BotFather) account.
2. Send `/revoke`, select `@CS2PredictorPublisherBot`, and confirm revocation.
3. If BotFather does not return a replacement immediately, send `/token`, select
   the same bot, and generate a new token.
4. Copy `.env.example` to `.env` and replace only
   `PASTE_THE_NEWLY_ROTATED_TOKEN_HERE` with the new token.
5. Never paste the replacement into chat, source code, test output, a command
   argument, or Git. `TELEGRAM/.env` is intentionally ignored.

Revocation is mandatory even if the original conversation was private: anyone
holding a Bot API token can operate that bot.

## One-time Telegram configuration

The channel must remain a broadcast-only channel:

1. In `CS2 Predictor | Daily Picks`, open **Manage Channel → Administrators**.
2. Add `@CS2PredictorPublisherBot` as an administrator.
3. Enable only **Post Messages**. Disable deletion, channel editing, inviting
   users, stories, and administrator management.
4. Do not link a **Discussion Group**. Without one, subscribers can read the
   channel but cannot write or comment.

## Installation

From PowerShell:

```powershell
Set-Location C:\dev\CS2-Predictor\TELEGRAM
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally only after rotating the exposed token:

```dotenv
TELEGRAM_BOT_TOKEN=<new token from BotFather>
TELEGRAM_CHANNEL_ID=@cs2DailyPicks
TELEGRAM_STATE_DB_PATH=data/telegram_publish_state.sqlite3
TELEGRAM_REQUEST_TIMEOUT_SECONDS=15
```

Relative paths in `.env` are resolved below `TELEGRAM/`, independently of the
shell's current directory.

## Usage

Always preview a date first. A dry run prints both posts and the exact report
preview, but makes no Telegram request, opens no publication database, changes
no publication state, and does not write `daily_report.txt`:

```powershell
.\run_telegram.ps1 -Date 2026-08-11 -DryRun
```

Publish both messages and generate the report:

```powershell
.\run_telegram.ps1 -Date 2026-08-11
```

Omit `-Date` to use today's date in `Europe/Madrid`, independently of the
machine timezone:

```powershell
.\run_telegram.ps1 -DryRun
.\run_telegram.ps1
```

`run_telegram.ps1` is safe to invoke from any working directory. It uses
`TELEGRAM/.venv` when available and otherwise searches for `py -3` and then
`python`. The equivalent direct Python interface is:

```powershell
.\.venv\Scripts\python.exe scripts\publish_cs2.py --date 2026-08-11 --dry-run
```

The publisher records successful sends in
`TELEGRAM/data/telegram_publish_state.sqlite3`. Re-running the same source run and date
skips messages already confirmed by Telegram, preventing accidental duplicate
posts. An unconfirmed or failed attempt is blocked for manual reconciliation;
the publisher never retries an ambiguous delivery automatically. The report is
written only after both publication slots are confirmed. An idempotent rerun
whose two slots are already confirmed safely regenerates it; a publication
failure leaves the previous report untouched.

## Message shape

The first post follows this shape:

```text
🔥 BEST OPPORTUNITIES

🎮 Match: Team One vs Team Two
🏆 Winner: Team One
📊 Win probability: 71.89%
Confidence: HIGH

⚠️ Model estimate — not a guarantee.
```

The second post uses the same fields for every remaining match under
`📋 TODAY'S OTHER PICKS`. Telegram's 4096-character limit is enforced before
network access. Input validation rejects absent teams, inconsistent winners,
invalid probabilities, duplicate matches, or an ambiguous daily source instead
of publishing invented values.

## Incremental opportunities report

`daily_report.txt` is UTF-8 text intended for copying into another publication
workflow. It contains no emojis and includes every not-yet-exported qualified
opportunity in the latest run from the requested date onward. Its shape is:

```text
CS2 BEST OPPORTUNITIES
Date: 2026-08-11

URL: https://www.hltv.org/matches/<match-id>/<match-slug>
Winner: Team One
Match: Team One vs Team Two
Win probability: 71.89%
Confidence: HIGH

This prediction was calculated using my predictive Machine Learning and Artificial Intelligence model as part of my Data Science PhD thesis.
The probability is a model estimate rather than a guaranteed outcome.

See all model predictions for free on Telegram:
@cs2DailyPicks
```

Each URL must be a validated canonical HTTPS match URL on `hltv.org`; missing or
ambiguous source data fails explicitly rather than inventing a link. The report
CTA deliberately uses only `@cs2DailyPicks`, without a `t.me` URL. SQLite stores
every exported match ID permanently; when there are no unseen opportunities, the
file contains an explicit status instead of repeating a previous match.

## Root orchestrator

Running `C:\dev\CS2-Predictor\start.ps1` first executes CS2. If CS2 succeeds,
the root orchestrator then calls `TELEGRAM/run_telegram.ps1`; TENNIS runs
afterwards even if Telegram fails. `-DryRun` and `-WhatIf` omit both Telegram
and TENNIS. `-Retrain` is forwarded only to CS2 and TENNIS, never Telegram.

The final exit code prioritizes a TENNIS failure; when TENNIS succeeds, it
returns the Telegram exit code. This keeps a messaging failure visible without
preventing the independent tennis workflow.

## Tests

Tests are local and do not contact Telegram:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

They cover source validation, formatting, report URL/content and atomic
replacement, transport with an injected fake HTTP adapter, idempotency,
Europe/Madrid date selection, dry-run side-effect isolation,
documentation/launcher invariants, and secret scanning.

`requirements.txt` declares pytest, Ruff and mypy as direct tooling requirements,
plus `tzdata` for Europe/Madrid on Windows. Runtime uses CPython 3.13 and has no
third-party Python imports. No requirements lock exists for this component.
Lint/type checks can be run with `python -m ruff check --no-cache src scripts
tests` and `python -m mypy --cache-dir .pytest_cache/mypy --follow-imports normal
src scripts`. The real entrypoint smoke is `run_telegram.ps1 -DryRun`; it never
loads credentials or writes publication state.

### Verification on 2026-09-05

- CPython 3.13.15, installation from this component's requirements and `pip check`: passed.
- Ruff lint/format across src/scripts/tests, mypy across 10 runtime files:
  passed; **69 offline tests** include import coverage, threshold boundaries,
  confidence validation, append-future stability and idempotency.
- Real `run_telegram.ps1 -Date 2026-09-05 -DryRun`: one opportunity
  (ALKA, 66.10%, MEDIUM) and 22 other picks; the report preview has eight
  opportunities including future dates. No Telegram requests were made.
- SQLite publication state and the existing daily report have identical
  SHA-256 hashes before/after the dry run. No CS2 model, ledger or features
  were modified.
- Architecture review: publication-only filter, domain eligibility retained.
  Anti-leakage review: uses frozen exported values without recalculation.
  Dependency review: isolated Telegram environment, declared tools/timezone
  data and actual launcher verified.

## Troubleshooting

- **Unauthorized / 401:** revoke the token in BotFather and place the newly
  generated value in `.env`; confirm there are no quotes or spaces around it.
- **Forbidden / 403:** confirm the bot is still a channel administrator with
  **Post Messages** enabled.
- **Chat not found:** keep `TELEGRAM_CHANNEL_ID=@cs2DailyPicks`; private channels
  require their numeric chat ID instead.
- **No prediction source for the date:** run the CS2 predictor first. The
  Telegram component deliberately fails instead of reusing a different date.
- **Already published:** the state database found a confirmed send for the same
  date, source run, and message slot. This is expected idempotent behaviour.
