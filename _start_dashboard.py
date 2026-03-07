"""
Start just the dashboard server — no demo data.
Point PIXELTABLE_HOME to whichever DB you want to browse.

Usage:
  PIXELTABLE_HOME=/tmp/pxt_pipeline_demo python _start_dashboard.py
  # or just:
  python _start_dashboard.py   (uses ~/.pixeltable by default)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

db_dir = os.environ.get('PIXELTABLE_HOME', '')
if db_dir:
    print(f'Using PIXELTABLE_HOME={db_dir}')
else:
    print('Using default PIXELTABLE_HOME (~/.pixeltable)')

import pixeltable as pxt

pxt.init()

print(f'\nDashboard: http://localhost:8080')
print('Ctrl+C to stop\n')

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('\nStopped.')
