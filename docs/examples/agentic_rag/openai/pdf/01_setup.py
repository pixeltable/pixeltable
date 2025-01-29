from config import DIRECTORY

import pixeltable as pxt

# Create fresh environment
pxt.drop_dir(DIRECTORY, force=True)
pxt.create_dir(DIRECTORY, if_exists='ignore')

scripts = [
    '02_create_pdf_index.py',
    '03_create_agent.py',
]

for script in scripts:
    print(f'\nExecuting {script}...')
    exec(open(script).read())
