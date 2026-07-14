import os
import sys

# Guarantee the local `tests` package wins over any `tests/` a rogue wheel
# dumped into site-packages, regardless of cwd / -m / .pth ordering.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pixeltable as pxt
from tests.utils import local_embedding

if __name__ == '__main__':
    print('===== RUNNING setup_postgres_test.py =====')
    pxt.init()
    t: pxt.Table = pxt.create_table('tbl', {'id': pxt.Int, 'string': pxt.String})
    t.add_embedding_index(t.string, embedding=local_embedding)
    t.insert({'id': n, 'string': f'This is a whole bunch of text, sentence number {n}'} for n in range(10))
