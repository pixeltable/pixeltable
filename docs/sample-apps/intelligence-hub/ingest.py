"""Data ingestion for the Intelligence Hub.

Run after setup_pixeltable.py to seed data from multiple sources:
    python ingest.py

Re-run at any time to add more data -- all computed columns
(summarization, scoring, notifications, exports) execute automatically.
"""

import os
from datetime import datetime, timezone

import pandas as pd

import pixeltable as pxt

import config
import custom_udfs.google_sheets as google_sheets

sources = pxt.get_table(f'{config.APP_NAMESPACE}.sources')


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── Source A: Web URLs ────────────────────────────────────────────────────────
# Insert URLs directly -- Pixeltable fetches + parses HTML via pxt.Document.

for item in config.SEED_URLS:
    sources.insert([{
        'url': item['url'],
        'title': item['title'],
        'doc': item['url'],
        'origin': 'web',
        'metadata': {},
        'timestamp': _now(),
    }], on_error='ignore')
    print(f'  Web: inserted "{item["title"]}"')


# ── Source B: Google Sheets (optional) ────────────────────────────────────────
# Reads rows from a Google Sheet and maps them into the sources table.

if config.GOOGLE_SHEETS_CREDENTIALS and config.GOOGLE_SHEET_ID:
    try:
        sheet_rows = google_sheets.import_rows(
            config.GOOGLE_SHEETS_CREDENTIALS, config.GOOGLE_SHEET_ID,
        )
        mapped = [
            {
                'url': row.get('url', ''),
                'title': row.get('title', ''),
                'doc': row.get('url', ''),
                'origin': 'sheet',
                'metadata': {k: v for k, v in row.items() if k not in ('url', 'title')},
                'timestamp': _now(),
            }
            for row in sheet_rows
            if row.get('url')
        ]
        if mapped:
            sources.insert(mapped, on_error='ignore')
        print(f'  Google Sheets: inserted {len(mapped)} rows')
    except Exception as e:
        print(f'  Google Sheets: skipped ({e})')
else:
    print('  Google Sheets: not configured')


# ── Source C: Local CSV ───────────────────────────────────────────────────────
# Reads sample_sources.csv and maps each row into the sources table.

sample_csv = os.path.join(os.path.dirname(__file__), 'sample_sources.csv')
if os.path.exists(sample_csv):
    df = pd.read_csv(sample_csv)
    csv_rows = [
        {
            'url': row.get('url', ''),
            'title': row.get('title', ''),
            'doc': row.get('url', ''),
            'origin': 'csv',
            'metadata': {},
            'timestamp': _now(),
        }
        for row in df.to_dict('records')
        if row.get('url')
    ]
    for row in csv_rows:
        sources.insert([row], on_error='ignore')
    print(f'  CSV: inserted {len(csv_rows)} rows')
else:
    print('  CSV: no sample_sources.csv found')


# ── Summary ───────────────────────────────────────────────────────────────────

total = sources.count()
print(f'\nIngestion complete. {total} total sources in table.')
print('All computed columns are now executing (summary, scoring, notifications).')
print()
print('To add more data later:')
print()
print('  import pixeltable as pxt')
print(f'  sources = pxt.get_table("{config.APP_NAMESPACE}.sources")')
print('  sources.insert([{')
print('      "url": "https://example.com/article",')
print('      "title": "My Article",')
print('      "doc": "https://example.com/article",')
print('      "origin": "manual",')
print('      "metadata": {},')
print('  }])')
