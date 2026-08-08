# A-share fine-tuning data preparation

`data/prepare_a_share_finetune.py` builds a small daily dataset for Kronos from
currently eligible A shares. `--provider auto` attempts Adata first, then
AkShare Eastmoney history, then AkShare's Tencent history route. Live
eligibility is always established by Adata or AkShare; each downloaded file
records the history route that actually succeeded.

The AkShare path fetches:

- current CSI 300 constituents from CSIndex (`ak.index_stock_cons_csindex("000300")`),
  with the older Sina route as a fallback;
- the current A-share code/name table (`ak.stock_info_a_code_name()`);
- daily history from Eastmoney (`ak.stock_zh_a_hist`) or Tencent
  (`ak.stock_zh_a_hist_tx`).

The allowed universe is the intersection with the live listed-code table and
either CSI 300 membership or the board prefixes `300`/`301` (ChiNext) and
`688`/`689` (STAR Market). A fixed preference list only expresses sector-leader
ordering; it is never treated as proof of eligibility. The default selection is
10 symbols, taking one preferred symbol per broad sector before using fallbacks.

## Run

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install adata akshare pandas
.\.venv\Scripts\python.exe data\prepare_a_share_finetune.py
```

Useful options:

```powershell
.\.venv\Scripts\python.exe data\prepare_a_share_finetune.py --provider auto --dry-run
.\.venv\Scripts\python.exe data\prepare_a_share_finetune.py --provider adata --dry-run
.\.venv\Scripts\python.exe data\prepare_a_share_finetune.py --provider akshare_tencent --adjust ''
.\.venv\Scripts\python.exe data\prepare_a_share_finetune.py --count 6 --start-date 20150101
.\.venv\Scripts\python.exe data\prepare_a_share_finetune.py --symbols 600036,300750,688981
```

`--symbols` does not bypass the live eligibility check. The pipeline has no
offline/stale-symbol fallback: if all selected providers fail, it exits
non-zero with the failure and does not write partial files. The Adata route
currently supports qfq-compatible history only. The Tencent route supports
raw/qfq/hfq through AkShare; validate qfq output for long histories because a
provider can emit invalid adjusted prices. The providers are retried with an
increasing delay; tune this with
--retries and --request-delay when a public endpoint is rate-limited.

## Outputs

After every selected symbol has passed validation, the script writes this
directory:

```text
data/a_share_finetune/
├── <symbol>.csv
└── manifest.json
```

Each symbol CSV has exactly these columns, in this order:

```text
timestamps,open,high,low,close,volume,amount
```

The requested adjustment is recorded in the manifest. Validation checks
missing/non-numeric values, unique sorted dates, requested date bounds, weekday
dates, positive OHLC values, OHLC high/low invariants, non-negative volume and
amount, and a minimum of 512 daily rows per symbol. The manifest records the
provider/version, live eligibility source for each symbol, selection rationale,
row/date ranges, and SHA-256 of each CSV.

## Build a local fine-tuning dataset

For an offline smoke test, the repository includes 28 cached STAR Market
series under `data/direction_universe`. They are explicitly treated as STAR
Market data only; they are not asserted to be current CSI 300 constituents.

```powershell
.\.venv\Scripts\python.exe data\build_local_finetune_dataset.py
```

This writes `data/a_share_finetune_local/` with per-symbol CSVs,
`train_data.pkl`, `val_data.pkl`, `test_data.pkl`, and a manifest. The default
consumer configuration is strict `128 -> 10` and checks that manifest before
loading samples. For a live CSI300/创业板/科创板 dataset, run the online
preparation command first and pass its output directory to the local builder:

```powershell
.\.venv\Scripts\python.exe data\build_local_finetune_dataset.py `
  --source-dir data\a_share_finetune
```
