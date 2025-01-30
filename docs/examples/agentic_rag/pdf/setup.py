import os

from config import DIRECTORY

import pixeltable as pxt

# Create fresh environment
pxt.drop_dir(DIRECTORY, force=True)
pxt.create_dir(DIRECTORY, if_exists='ignore')

tables = [
    os.path.join('tables', 'create_pdf_index.py'),
    os.path.join('tables', 'create_agent.py'),
]

for table in tables:
    print(f'\nExecuting {table}...')
    exec(open(table).read())

