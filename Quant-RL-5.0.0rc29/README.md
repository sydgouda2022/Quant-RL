# Quant-RL 5.0.0rc29

MT5-only, single-position reinforcement-learning forex research and execution.
This is **not** the older Quant project. It uses the MetaTrader 5 terminal currently
logged in, with no CSV, synthetic, Yahoo, or offline historical-data fallback.
LIVE is Demo-only by default and requires a verified broker IANA timezone.

## Run it

Use **64-bit Python 3.12** on **Windows**, with a fresh environment and the
MetaTrader 5 desktop terminal installed, connected, and authenticated.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python main.py
```

Alternatively, run `main.py` from an IDE. No command-line arguments are
required; it automatically chooses RESEARCH or LIVE from MT5 conditions.
**Safety gate:** LIVE refuses funded accounts by default and also refuses to
start until `broker_connection.mt5_server_timezone` is set to the verified
IANA timezone used by the broker. Keep `live.allow_non_demo_account: false`
for Demo operation. Setting it to `true` is an explicit opt-in that permits
funded-account LIVE. No Windows/demo end-to-end certification has been performed.

If the terminal is already logged in, keep `broker_connection.mt5_login: 0`.
For other installations use `QUANT_MT5_LOGIN`, `QUANT_MT5_PASSWORD`, and
`QUANT_MT5_SERVER` environment variables, not hardcoded credentials.

## Operating decisions

- One configured symbol is researched at a time; models belong to symbols and
  broker economics, and the champion may seed subsequent models when compatible.
- Weekly FX schedule uses an explicitly configured IANA MT5 server timezone:
  block new entries from Friday 23:00 through Sunday 21:00 by default. Both
  hours are configurable in `execution.friday_cutoff_server_hour` and
  `execution.sunday_open_server_hour`. Set
  `broker_connection.mt5_server_timezone` only after verifying the broker's
  server timezone. `null` blocks LIVE. Python's timezone database handles DST.
- MT5 bar, tick and deal timestamps are kept exactly in the UTC epoch returned
  by the MT5 Python API. Broker-local timezone conversion is used only for the
  weekly schedule; it is never applied to market data or trade history.
- LIVE is Demo-only by default. `live.allow_non_demo_account: true` is a
  separate explicit operator opt-in for funded-account operation.
- The weekly block prevents new entries; it does not forcibly close broker-held
  positions or replace their normal broker-side SL/TP management.
- Market closed by the weekly schedule: RESEARCH eligible MT5 symbols sequentially.
- Market open, an eligible configured symbol lacks a certified model: RESEARCH
  missing models sequentially.
- Market open, models qualified: LIVE a selected eligible model, with **one
  position maximum** and no simultaneous research process.
- Quote stale, broker history insufficient, model stale, or account changed:
  stop or veto safely, never substitute fabricated data.
- Research fetches new, completed H1 and M1 observations **directly from MT5 on
  each run**, selects a rolling four-calendar-year window, and constructs H4
  observations by default. An in-memory immutable dataset receives a content
  fingerprint and integrity/coverage manifest. **No duplicate historical
  `.npz` archives, CSV cache, or local historical database is created.**

Research uses 50% training, 25% validation (split between search and promotion),
and 25% held-out evaluation. Promotion finalists are tested sequentially on
that held-out segment as a fallback selection rule; once used this way, it is
not an independent final test. Independent confirmation requires a fresh future
window. Models are published only after the configured qualification checks.
Rolling tests do not guarantee future performance.

## Training and policy

- PyTorch policy, PPO training and Optuna search operate sequentially per symbol.
- Policy actions: BUY / SELL / NO_TRADE, stop distance, target distance and
  risk allocation. The 26-check examiner cannot be bypassed by a learned policy.
- Training receives simulated historical net trading outcomes, including
  explicitly labeled research-only spread, commission, slippage and financing
  **assumptions**. NO_TRADE/veto penalties affect reward, **not** MT5 cash.
- Requested configuration defaults remain **1,000 trials × 1,000 PPO episodes
  × 1,000 PPO epochs** (upper bounds, not a promise of completion). This can
  be extremely expensive. Optuna pruning, PPO KL stopping and a 7,200-second
  research session budget reduce unnecessary work.
- Optuna studies, per-episode recovery checkpoints, accepted model weights,
  model manifest and trade/evaluation reports are persisted. They are **not**
  historical market-data inputs. The application re-fetches MT5 data anyway.

The policy's variable-length, mixed continuous/discrete trading output,
custom economic replay and recovery format are tightly coupled to the existing
PyTorch/Optuna implementation. Replacing it with a generic PPO package without
adapting these contracts would not be a safe cleanup. No extra RL framework
has been added solely to duplicate existing working features.

## What MT5 supplies directly

| Responsibility | Source |
| --- | --- |
| Prices, history and broker symbol specifications | MT5 native Python API |
| Account currency, balance, equity, leverage and positions | `account_info`, `positions_get` |
| LIVE stop-loss economic estimate | `order_calc_profit` |
| LIVE margin estimate | `order_calc_margin` |
| Pre-submission validation and actual order | `order_check`, `order_send` |
| Executed deal cash, commissions and swaps | `history_deals_get` |

The research simulator remains necessary for policy training over historical
candles. MT5's **current** `order_calc_profit` is not an authenticated
historical commission, financing or FX-conversion model. The LIVE examiner's
predictive risk/cost checks stay as conservative guards; **the broker's final
profit/margin preflight is authoritative before submission**. Actual trades,
profit, commission, swaps and fees come from MT5 deal records, not an invented
LIVE ledger.

## Configuration

Edit `config.yaml`. It lists only the operator-relevant settings; all other
values have strict Pydantic defaults in `quant_rl/config/schema.py`. Removing
unused options is intentional: `data.storage_dir` and the cross-symbol
correlation parameters do not exist. No compatibility/migration aliases.

Changing the policy or economic rules invalidates old model eligibility. It
requires fresh MT5 research and requalification; do not force an old model
into LIVE. Research uses the current MT5 broker-server and account-currency identity.
MT5 API timestamps remain UTC. The IANA server timezone is used only to
interpret the weekly entry schedule; never shift historical bars, ticks or deals.

## LIVE safety and recovery

- Kill switch: set `QUANT_RL_KILL_SWITCH=1` or write
  `live_state/kill_switch.json` with `{"engaged": true}`.
- Manually managed economic blackout: set `QUANT_RL_ECONOMIC_BLACKOUT=1` or
  `live_state/economic_blackout.json` with `{"active": true}`. Neither MT5 nor
  this app supplies a trusted historical economic-news calendar.
- Missing/malformed safety state fails closed. The kill switch, broker
  connection and blackout are checked again immediately before sending.
- Broker-held positions/orders block new proposals. An uncertain order has a
  durable write-ahead journal; never retry it blindly. Reconcile against
  actual MT5 order, position and deal identities or stop for manual review.
- C14 consecutive-loss monitoring is broker-backed and bound to login,
  server, symbol and magic. An older or inconsistent checkpoint must be
  reconciled, not deleted to bypass a refusal.
- Kernel-managed `research.lock` prevents concurrent research/LIVE processes.
  Logs are local JSONL with secret redaction, size rotation and retention.

## Verify

```powershell
python -m pytest -q
```

The automated suite uses fake broker data **only in tests**; no real trades are
submitted during CI. See `BROKER_ACCEPTANCE_CHECKLIST.md` for the required
Windows MetaQuotes-Demo integration tests. Historic currency conversion,
broker-specific commission and financing models, partial-position accounting,
real trading-session/DST rules, four-year MT5 memory benchmarking and actual
order-recovery behavior are **not certified**. Do not assume funded LIVE
readiness or profitability from a green unit-test suite.
