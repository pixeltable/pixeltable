"""Insert a few online PDFs to test document rendering in the dashboard."""
import pixeltable as pxt

DIR = 'doc_demo'

if DIR in pxt.list_dirs():
    pxt.drop_dir(DIR, force=True)
pxt.create_dir(DIR)

t = pxt.create_table(f'{DIR}/papers', {
    'title': pxt.String,
    'doc': pxt.Document,
    'category': pxt.String,
})

rows = [
    {
        'title': 'Attention Is All You Need',
        'doc': 'https://arxiv.org/pdf/2312.00752',
        'category': 'ML',
    },
    {
        'title': 'US Constitution',
        'doc': 'https://www.archives.gov/files/founding-docs/constitution-page1-high-res.pdf',
        'category': 'Legal',
    },
    {
        'title': 'Python PEP 8',
        'doc': 'https://peps.python.org/pep-0008/',
        'category': 'Programming',
    },
]

t.insert(rows, on_error='ignore')
print(f'Inserted {t.count()} docs into {DIR}/papers')
print('Dashboard: http://localhost:8080')
