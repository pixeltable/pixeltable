"""
Insert 5000 images into a table to test dashboard scale/pagination.
Uses picsum.photos with varied IDs for distinct thumbnails.
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import pixeltable as pxt

pxt.init()

DIR = 'scale_test'
try:
    pxt.drop_dir(DIR, force=True)
except Exception:
    pass
pxt.create_dir(DIR)

t = pxt.create_table(
    f'{DIR}/images',
    {'label': pxt.String, 'score': pxt.Float, 'category': pxt.String, 'batch': pxt.Int, 'image_url': pxt.String},
)

labels = ['cat', 'dog', 'car', 'tree', 'building', 'person', 'bicycle', 'bird', 'flower', 'mountain']
categories = ['nature', 'urban', 'animals', 'vehicles', 'people']

TOTAL = 10_000_000
BATCH_SIZE = 10_000

for b in range(0, TOTAL, BATCH_SIZE):
    rows = []
    for i in range(b, min(b + BATCH_SIZE, TOTAL)):
        rows.append(
            {
                'image_url': f'https://picsum.photos/seed/{i}/200/200',
                'label': labels[i % len(labels)],
                'score': round((i * 7 % 100) / 100, 2),
                'category': categories[i % len(categories)],
                'batch': b // BATCH_SIZE,
            }
        )
    t.insert(rows)
    done = min(b + BATCH_SIZE, TOTAL)
    print(f'  {done:,}/{TOTAL:,} ({done * 100 // TOTAL}%)')

print(f'\nDone: {DIR}/images has {TOTAL} rows')
print('Dashboard: http://localhost:8080')
