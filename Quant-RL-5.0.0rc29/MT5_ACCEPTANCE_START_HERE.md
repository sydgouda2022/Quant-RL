# Quant-RL rc28 — MT5 Demo acceptance: first stage

This first stage is **read-only**. It does not call `order_check`, `order_send`, modify orders, or close positions.

## Before running

1. Use the Windows machine that runs Quant-RL.
2. Open MetaTrader 5 and sign in to the intended **MetaQuotes-Demo** account.
3. Confirm the terminal is connected and wait until quotes appear for the configured symbols.
4. Do not start `main.py` for this check. The application is Demo-only by default, but after you explicitly configure a verified IANA server timezone and the terminal reports Demo mode, the router may select LIVE.

## Run

From the extracted project folder, in the same Python environment used by Quant-RL:

```powershell
python -m pip install -e .
python tools/mt5_readonly_preflight.py
```

The report checks terminal/account connectivity, a masked account identifier, account currency, MT5 account trade mode and an explicit `account_is_demo` result, symbol specifications, UTC tick age, configured server timezone, and a small sample of completed H1/M1 bars for each configured symbol.

## Send back

Share the JSON output after reviewing it for broker/server details you do not want to disclose. Never share passwords, access tokens, or full account numbers.

## What this does not certify

This is only a connectivity and data smoke test. It does not prove four-year H1/M1 coverage, historical costs, DST/session correctness, order execution quality, crash recovery, or funded-account readiness. Those require separate controlled stages. No trade test should begin until the account identity and demo status are independently confirmed.
