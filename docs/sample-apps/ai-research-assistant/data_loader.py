from typing import List
import pixeltable as pxt
from pathlib import Path

def setup_document_table():
    """Create and initialize documents table."""
    # Constants
    BASE_URL = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/'
    DOCUMENT_NAMES = [
        'Argus-Market-Digest-June-2024.pdf',
        'Zacks-Nvidia-Repeport.pdf'
    ]

    # Initialize Pixeltable
    pxt.drop_dir('research', force=True)
    pxt.create_dir('research')

    # Create and populate table
    docs_table = pxt.create_table(
        'research.documents',
        {'document': pxt.Document}
    )

    docs_table.insert({'document': BASE_URL + doc} for doc in DOCUMENT_NAMES)
    return docs_table