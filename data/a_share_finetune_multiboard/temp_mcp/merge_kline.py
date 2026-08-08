import json
import re
import os
import csv
from collections import defaultdict

TRACKING_FILE = r"d:\workspace\Kronos\data\a_share_finetune_multiboard\temp_mcp\tracking.csv"
CSV_DIR = r"d:\workspace\Kronos\data\a_share_finetune_multiboard\csv"
MIN_DATE = "2010-01-01"

# Parse tracking file - use regex to extract all entries regardless of line formatting
with open(TRACKING_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern: stockcode(6 digits),setcode(1 digit),page(1-2 digits),filepath
pattern = r'(\d{6}),(\d),(\d+),(C:\\Users\\87521\\AppData\\Local\\Temp\\trae\\toolcall-output\\[a-f0-9-]+\.txt)'
matches = re.findall(pattern, content)

# Group by stock code
stock_files = defaultdict(list)  # {stock: [(page, filepath), ...]}
for stock, setcode, page, filepath in matches:
    stock_files[stock].append((int(page), filepath))

# Sort pages for each stock
for stock in stock_files:
    stock_files[stock].sort(key=lambda x: x[0])

print(f"Found {len(stock_files)} stocks in tracking file")
for stock in sorted(stock_files.keys()):
    print(f"  {stock}: {len(stock_files[stock])} pages")


def parse_temp_file(filepath):
    """Parse a temp file and return list of row dicts from Rows array."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # The file contains: "The MCP server responded with: [{...}]"
    # Find the JSON array
    json_start = raw.find("[{")
    if json_start == -1:
        return []

    # Extract the text field content
    # Find "详细K线数据:\n" then extract the JSON object after it
    text_marker = "详细K线数据:\\n"
    # The text field has escaped newlines as \\n and escaped quotes as \\\"
    # But in the raw file, it might be different

    # Try to find the marker with escaped newline
    idx = raw.find(text_marker)
    if idx == -1:
        # Try without escape
        text_marker = "详细K线数据:\n"
        idx = raw.find(text_marker)
    if idx == -1:
        # Try just "详细K线数据:"
        text_marker = "详细K线数据:"
        idx = raw.find(text_marker)

    if idx == -1:
        print(f"  WARNING: Could not find '详细K线数据' in {filepath}")
        return []

    # Find the JSON object start (first { after the marker)
    json_obj_start = raw.find("{", idx)
    if json_obj_start == -1:
        return []

    # The JSON object is embedded in a JSON string, so quotes are escaped as \"
    # and newlines as \n
    # We need to extract the raw text and unescape it

    # Find the matching closing brace by counting braces
    # But we need to account for escaped quotes
    depth = 0
    i = json_obj_start
    end = -1
    in_string = False
    escape = False

    while i < len(raw):
        c = raw[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == '\\':
            escape = True
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            i += 1
            continue
        if not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        i += 1

    if end == -1:
        print(f"  WARNING: Could not find matching brace in {filepath}")
        return []

    json_str_escaped = raw[json_obj_start:end]

    # Unescape: replace \" with " and \n with newline
    json_str = json_str_escaped.replace('\\"', '"').replace('\\n', '\n')

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error in {filepath}: {e}")
        return []

    rows = data.get("Rows", [])
    if not rows:
        return []

    return rows


def transform_row(row):
    """Transform a raw K-line row to CSV format."""
    # Data is YYYYMMDD, convert to YYYY-MM-DD
    data = row.get("Data", "")
    if len(data) != 8:
        return None
    timestamp = f"{data[:4]}-{data[4:6]}-{data[6:8]}"

    try:
        open_price = float(row.get("Open", 0))
        high = float(row.get("High", 0))
        low = float(row.get("Low", 0))
        close = float(row.get("Close", 0))
        amount = float(row.get("Amount", 0))
        raw_volume = row.get("RawVolume", 0)
        volume = int(float(raw_volume))
    except (ValueError, TypeError):
        return None

    return {
        "timestamps": timestamp,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    }


def read_existing_csv(filepath):
    """Read existing CSV and return list of row dicts."""
    rows = []
    if not os.path.exists(filepath):
        return rows
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                transformed = {
                    "timestamps": row["timestamps"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(float(row["volume"])),
                    "amount": float(row["amount"]),
                }
                rows.append(transformed)
            except (ValueError, KeyError):
                continue
    return rows


def process_stock(stock, pages):
    """Process a single stock: fetch new data, merge with existing, save."""
    print(f"\n--- Processing {stock} ({len(pages)} pages) ---")

    # Collect all new rows from temp files
    new_rows = []
    for page_num, filepath in pages:
        if not os.path.exists(filepath):
            print(f"  Page {page_num}: file not found {filepath}")
            continue
        rows = parse_temp_file(filepath)
        if rows:
            print(f"  Page {page_num}: {len(rows)} rows, first date: {rows[0].get('Data', 'N/A')}")
        else:
            print(f"  Page {page_num}: 0 rows (empty)")
        for row in rows:
            transformed = transform_row(row)
            if transformed:
                new_rows.append(transformed)

    # Read existing CSV
    csv_path = os.path.join(CSV_DIR, f"{stock}.csv")
    existing_rows = read_existing_csv(csv_path)
    print(f"  Existing CSV: {len(existing_rows)} rows")

    # Merge: combine and deduplicate by date
    all_rows = {}
    for row in existing_rows:
        all_rows[row["timestamps"]] = row
    for row in new_rows:
        # New data takes precedence (overwrites existing on same date)
        all_rows[row["timestamps"]] = row

    # Convert to list and sort by date
    merged = list(all_rows.values())
    merged.sort(key=lambda x: x["timestamps"])

    # Filter: date >= 2010-01-01, volume > 0, close > 0
    filtered = [
        row for row in merged
        if row["timestamps"] >= MIN_DATE
        and row["volume"] > 0
        and row["close"] > 0
    ]

    # Save
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamps", "open", "high", "low", "close", "volume", "amount"]
        )
        writer.writeheader()
        for row in filtered:
            writer.writerow(row)

    if filtered:
        print(f"  Saved: {len(filtered)} rows, date range: {filtered[0]['timestamps']} ~ {filtered[-1]['timestamps']}")
    else:
        print(f"  Saved: 0 rows (no data after filtering)")

    return len(filtered), filtered[0]["timestamps"] if filtered else "N/A", filtered[-1]["timestamps"] if filtered else "N/A"


# Process all stocks
results = []
for stock in sorted(stock_files.keys()):
    count, start_date, end_date = process_stock(stock, stock_files[stock])
    results.append((stock, count, start_date, end_date))

# Print summary
print("\n" + "=" * 70)
print("汇总报告:")
print("=" * 70)
print(f"{'股票代码':<10} {'行数':<8} {'开始日期':<12} {'结束日期':<12}")
print("-" * 70)
for stock, count, start, end in results:
    print(f"{stock:<10} {count:<8} {start:<12} {end:<12}")
