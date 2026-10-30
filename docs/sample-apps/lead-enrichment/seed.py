"""Seed sample prospects into the leads table (run after `pxt schema update app.py leads`).

Usage:
    export TREG_TOKEN=treg_...
    export OPENAI_API_KEY=sk-...
    python seed.py leads
"""

from __future__ import annotations

import sys
import uuid

import pixeltable as pxt

SAMPLE_LEADS = [
    {'full_name': 'Pierre Brunelle', 'domain': 'pixeltable.com', 'role': 'Founder & CEO', 'stage': 'new'},
    {'full_name': 'Marcel Kornacker', 'domain': 'pixeltable.com', 'role': 'Co-Founder & CTO', 'stage': 'queued'},
    {'full_name': 'Aaron Siegel', 'domain': 'pixeltable.com', 'role': 'Co-Founder', 'stage': 'new'},
]


def _lead_id(row: dict[str, str]) -> uuid.UUID:
    # Stable across reruns: a retry after a rolled-back insert reuses the treg Idempotency-Keys instead of paying again.
    return uuid.uuid5(uuid.NAMESPACE_URL, f'lead-enrichment-seed:{row["full_name"]}:{row["domain"]}')


def main(catalog: str = 'leads') -> None:
    t = pxt.get_table(f'{catalog}/leads')
    existing = {(r['full_name'], r['domain']) for r in t.select(t.full_name, t.domain).collect()}
    fresh = [r for r in SAMPLE_LEADS if (r['full_name'], r['domain']) not in existing]
    if not fresh:
        print(f'{catalog}/leads already has the sample prospects ({t.count()} rows)')
        return
    status = t.insert([{'lead_id': _lead_id(row), **row} for row in fresh], on_error='ignore')
    print(f'inserted {len(fresh)} prospects -> count={t.count()} errors={status.num_excs}')
    print(t.select(t.full_name, t.domain, t.email, t.verdict, t.served_by, t.tier, t.cost_usd).collect())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'leads')
