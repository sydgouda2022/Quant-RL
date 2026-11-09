# Quant-RL rc28: external acceptance gates (not yet passed)

This checklist describes tests requiring the operator's **Windows MT5 demo** environment. Unit tests never establish production market accuracy. Never put account passwords, access tokens or broker tickets in public logs or shared reports.

## 1. Platform and market evidence

- Confirm broker login, trading-server identity, deposit currency, instrument suffix, minimum/maximum lots, lot steps, price tick size, tick value, permitted trade direction and symbol filling modes directly from MT5.
- Confirm the selected MT5 terminal's `Max. bars in chart` supports the required history; measure actual four-year H1 and M1 calendar coverage, holiday gaps and missing-minute counts for each configured instrument.
- Run a fresh multi-year M1 acquisition for one symbol, observe peak RSS and CPU, and verify the in-memory dataset fingerprint and reported M1 coverage. Repeat after restart; history must be freshly acquired each run.
- Confirm that MT5 bar, tick and deal timestamps are UTC epochs and are not shifted. Independently identify the broker's IANA server timezone and DST behavior; configure `broker_connection.mt5_server_timezone` only after verification. Test Friday close, holiday sessions and Sunday reopen against real dated ticks and candles. The weekly schedule remains generic and does not model holidays or early closes.

## 2. Broker economics

- For long and short demo orders, compare broker `order_calc_profit`, `order_calc_margin`, fill-price execution, and the eventual closed-deal cash ledger. Include adverse tick movements and partial fills.
- Capture *actual* broker commissions (including entry- and exit-side charges), volume tiers, rollover/swap units, triple swaps, financing date shifts, and fees on a representative multi-day holding period. Research's configured cash-per-lot financing is a **proxy**, not an authenticated agreement.
- For a non-USD account, verify actual point-in-time conversion rates of each traded pair into the account currency for entry, mark-to-market and exit; the present historical engine does **not** provide that complete series.

## 3. LIVE recovery and operational risk

- In a demo account, simulate a terminal crash after order intent was journaled but before the response, and verify the pending order is resolved from authenticated positions/orders/deals without resubmission.
- Exercise kill switch, broker heartbeat loss, economic blackout and spread/quote widening immediately before submission. Confirm the MT5 broker-side order preflight and account identity match the research certificate.
- Validate C14 across entry commission, partial exit, close-by operations, netting reversals, corrections, weekend pauses, and reboots. The **deal-based** checkpoint is not a fully certified closed-position accounting engine.
- Existing nonzero-offset C14 checkpoint upgrades require reconciled, independently verified broker history before LIVE is re-enabled. Zero-offset UTC checkpoints are upgraded automatically; do not delete a checkpoint to avoid the safety refusal.

## 4. Research and deployment

- Verify the 50/25/25 chronology, split-isolated normalization, search/promotion validation separation, held-out monitoring, and multi-seed stability on real broker history.
- Observe wall-time budget, disk space, checkpoint restart, Optuna commit recovery and artifact promotion under interruptions. Avoid simultaneously launching a second RESEARCH/LIVE process.
- Compare modeled reports with actual demo trade statements and retain objective diagnostics before any funded deployment decision.

**Status: all external gates pending.**
