## 5.0.0rc28 — Phase 1 safety and timestamp contract

- Accept `WEEKLY_SCHEDULE` as a valid automatic research trigger so weekend
  routing hands off to the research session without raising `ValueError`.
- Make `quant_rl.config` exports lazy to remove the circular import between
  configuration schema, timeframe alignment and MT5 source modules.
- Enforce Demo-only LIVE by default in automatic routing, the MT5 broker adapter
  and the direct LIVE session. Funded LIVE requires explicit
  `live.allow_non_demo_account: true`; unknown account mode fails closed.
- Remove broker offsets from MT5 bar, tick and deal timestamps. They remain UTC
  epochs as returned by MT5. Use an explicitly verified IANA
  `mt5_server_timezone` only for the weekly schedule, with DST handled by
  `zoneinfo`. An unset timezone blocks LIVE.
- Apply identical M1 price-path and spread-coverage qualification to challengers
  and incumbents.
- Add Windows-compatible `tzdata` dependency for IANA timezone support.
- Offline and mocked-MT5 verification only; no terminal connection or broker
  order was used. Legacy nonzero-offset C14 checkpoints require reconciliation.

## 5.0.0rc27 — Generic Friday-night FX entry cutoff

- Replace the approximate weekend heuristic with one explicit weekly FX entry
  schedule evaluated in configured MT5 server time. Default: Friday 23:00
  cutoff; Sunday 21:00 reopen. Both hours are configurable.
- The router blocks LIVE and routes eligible symbols to RESEARCH during the
  scheduled weekend window even if the broker continues streaming quotes.
- The C5 pre-trade firewall uses the same schedule and the same verified
  server-minus-UTC offset as LIVE timestamp normalization. Sunday LIVE resumes
  only when MT5 is connected, quotes are fresh, and a qualified model exists.
- The schedule blocks new entries only. It does not force-close open positions
  or alter broker-side stop-loss/take-profit management.
- Weekday/holiday quote outages remain unclassified and fail closed; the
  schedule does not pretend to model broker holidays or early closes.
- Added tests for Friday cutoff boundaries, configured server offset, Saturday
  block, Sunday reopen, router behavior with fresh quotes, and C5 consistency.
- Validation: 222 automated tests pass. No live MT5 integration or broker
  session-calendar verification performed.

## 5.0.0rc26 — Conservative broker-session classification

- A broker-disabled `trade_mode` is no longer treated as proof that the
  market is closed. It may indicate account restrictions, symbol suspension,
  maintenance, or broker-side failure.
- Automatic routing classifies closure only under the existing conservative
  weekend/no-recent-tick heuristic. Weekday and holiday ambiguity blocks LIVE
  and is not mislabeled as a confirmed closure.
- Added regression coverage for weekend research routing and weekday
  broker-disabled ambiguity.
- Validation: 216 automated tests pass. No live MT5 integration or
  broker-session calendar verification performed.

## 5.0.0rc25 — Deployability certificate hardening

- Extend `policy_hash` to cover the rolling research lookback, abstention reward economics, the complete qualification standard, and live execution timing/blackout/deviation settings.
- Keep search-budget settings and deployment identity fields (symbol, login, magic) outside the policy hash because they do not redefine the learned policy contract.
- Increment the policy contract version so existing models are conservatively requalified under the expanded certificate.
- Add tests proving behavior/qualification changes invalidate model eligibility, while Optuna/PPO search budgets and deployment identifiers do not.
- Correct README wording: the sequential held-out fallback consumes that test segment for selection; independent confirmation requires a fresh future window.
- Validation: 215 automated tests pass. No live MT5 broker integration or real-money execution was performed.

# Release notes

## 5.0.0rc28 — Phase 1 safety and timestamp contract

- Accept `WEEKLY_SCHEDULE` as a valid automatic research trigger so weekend
  routing can hand off to the research session without raising `ValueError`.
- Make `quant_rl.config` exports lazy to remove the import cycle between
  configuration schema, timeframe alignment and MT5 source modules.
- Enforce Demo-only LIVE by default in both automatic routing and the direct
  MT5 broker/LIVE-session entry points. Funded LIVE requires the explicit
  `live.allow_non_demo_account: true` setting; unknown account mode fails closed.
- Remove broker offsets from MT5 bar, tick and deal timestamps. These remain
  UTC epochs as returned by the MT5 Python API. Use an explicitly verified
  IANA `mt5_server_timezone` only for the weekly broker-local schedule, with
  daylight-saving transitions handled by `zoneinfo`. An unset timezone blocks LIVE.
- Apply identical M1 price-path and spread-coverage qualification to challengers
  and incumbents.
- Add Windows-compatible `tzdata` dependency for IANA timezone support.
- Validation is limited to offline tests and mocked MT5 interfaces; no terminal
  connection or broker order was used. Existing nonzero-offset loss-history
  checkpoints require reconciliation before reuse.


## 5.0.0rc24 — UTC window semantics and architecture consolidation

- Anchor each four-calendar-year research interval to the exact UTC execution instant; only fully completed candles at or before that instant enter the dataset.
- Keep M1 intrabar evidence bounded by the close of the last completed base-timeframe candle, never by a still-forming candle.
- Move shared action contracts to `quant_rl.contracts`; remove the obsolete `intelligence/action_space.py` module after migrating all imports.
- Move broker instrument certification beside the authoritative `SymbolSpec` in `data/mt5_source.py`; remove the redundant `data/certificates.py` module.
- Consolidate candidate replay and qualification with `research/evolution.py`; remove `research/evaluation.py`.
- Derive config timeframe validation from `data/time_alignment.py`. Make `quant_rl.data` exports lazy to prevent import cycles while preserving package-level imports.
- Keep the immutable NumPy dataset arrays and public `bars` representation unchanged in this release; no risky data-model rewrite was made solely to reduce memory duplication.
- Validation: 208 automated tests pass. No live MT5 broker integration or real-money execution was performed.

## 5.0.0rc23 — Top-K tournament and abstention calibration

- Retain and recover the top 9 Optuna candidate snapshots for each current study.
- Adjudicate every finalist on the withheld promotion-validation segment and at 1.5x modeled costs.
- Test promotion-qualified finalists sequentially on the held-out segment; select the first with positive net return. The report explicitly marks this held-out segment as consumed for selection, not as an independent final test.
- If no finalist clears the held-out fallback, preserve the best promotion candidate for failure analysis without publishing it as a new model.
- Set the default and explicit configuration for abstention reference volume to 0.01 lots. The resulting cash penalty remains dependent on each symbol's point-in-time spread and account-currency pip value.
- Validation: 208 automated tests pass. Live MT5 broker acceptance and real-money execution have not been performed by this source-level change.

# Quant-RL 5.0.0rc23 — MT5-first streamlining

**Baseline:** rc21. **Scope:** delete redundant infrastructure; preserve
execution safety and economic RL behavior. No original Quant repository files
were modified.

## Deleted

- Unused cross-symbol correlation engine, type, and associated parameters.
  Neither RESEARCH nor LIVE consumed that data, and this application permits
  only one active position.
- Dormant `MarketDataSource` protocol and unused `economic_score` helper.
- Two unused `ModelPopulation.publish` parameters and their obsolete call sites.
- Redundant `.npz` H1/M1 history archive writer, manifest sidecar writer,
  retention sweeper, and `data.storage_dir` setting. This prevents extra
  multi-year disk writes for source data that the application never reads.
- Legacy `HistoricalStore` naming and `load_or_fetch` ambiguity. The single
  implementation is now `MT5History.fetch`; it always queries fresh MT5 data.
- Obsolete logger naming, unused imports, and a nonessential compatibility
  import in LIVE. Updated the test suite to the new direct interfaces.
- Excess default-configuration noise: `config.yaml` now exposes the key
  knobs without repeating every validated library default.

## Preserved

- Live broker-native `order_calc_profit`, `order_calc_margin`, `order_check`,
  `order_send`, account identity, and executed-deal history.
- Mandatory C1–C26 examiner, one-position constraint, order journal and
  broker-backed loss protection.
- Fresh MT5-only H1/M1, four-year rolling window and provenance fingerprint.
- Research simulator and transparent historical financing/commission estimates
  **only** where required for PPO learning and backtesting.
- Optuna, economic PPO, persistent trials, recovery checkpoints, per-symbol
  models and weighted performance reports.
- 1,000/1,000/1,000 requested upper-bound training configuration.

## Compatibility and risk

No old signatures or migrations are provided. Removed economic-policy fields
change the configuration hash, so a model from an earlier release must be
requalified before LIVE use. Model files, research history and local order
journals have not been deleted by this source-only packaging task. Do not
silently delete broker state on deployment.

Fake-broker tests prove code-path behavior, **not** Windows/MT5 order or
financing correctness. Historical conversions, real broker swap fees, partial
fills/position accounting, server DST/session timing, and real recovery
certification remain separate acceptance work.


## 5.0.0rc29 — Phase 2: M1 streaming and session deadline

- Added an incremental M1 iterator over bounded 14-day MT5 requests. The research path consumes minute bars as a stream and stores compact typed buffers before creating the immutable NumPy evidence arrays; it no longer builds a multi-year list of Python `OHLCVBar` objects. The explicit `fetch_m1_range` convenience method remains materializing for callers that request a list.
- M1 validation is performed as rows are consumed, preserving timestamp ordering, minute alignment, OHLC integrity, and spread validity checks. Fingerprint inputs and resulting evidence arrays retain the same semantics.
- Propagated the research session's monotonic deadline through MT5 M1 acquisition, M1 freezing, dataset preparation checkpoints, optimization completion, and candidate/ incumbent replay. The economic simulator checks the deadline between bars and aborts the session rather than proceeding to another symbol.
- Deadline checks cannot interrupt an individual blocking MT5 terminal API call or an atomic OS operation; the budget is enforced immediately after such calls return and between streamed requests / replay bars.
- No strategy, model architecture, risk, qualification thresholds, or optimization-trial budget was changed.
