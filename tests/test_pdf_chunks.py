import pixeltable as pxt
from pixeltable.type_system import ArrayType, Float
from pixeltable.exprs import ColumnRef

import fitz  # pymupdf
import requests
import os
from dotenv import load_dotenv
import numpy as np

from pixeltable.iterators import PdfPageExtractor

import os
from typing import List

from pixeltable.catalog import Catalog

# Drop table
pxt.drop_table('pdf_docs', if_not_exists="ignore", force=True)

# Drop view
pxt.drop_table('pdf_page_chunks', if_not_exists="ignore", force=True)

# get all PDF files from sample file folder (up to 50 : parameter limit)
def find_pdfs(path: str, limit: int = 50, recursive: bool = True) -> List[str]:
    print( "\nGather PDF files in path: ", path)
    pdf_paths = []

    if recursive:
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_paths.append(os.path.join(root, file))
                    if len(pdf_paths) >= limit:
                        return pdf_paths
    else:
        for file in os.listdir(path):
            if file.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(path, file))
                if len(pdf_paths) >= limit:
                    break

    return pdf_paths

# get some example PDF files from project
pdfs = find_pdfs("tests/data/documents", limit=50)
print( "\nINGEST the following files:" , pdfs, "\n" )

# Create a table
docs = pxt.create_table('pdf_docs', {'doc': pxt.String}, if_exists='replace_force')

# Create chunks view with image per page
chunks_view = pxt.create_view(
    'pdf_page_chunks',
    docs,
    iterator=PdfPageExtractor.create(document=docs.doc)
)

for doc_path in pdfs:
    # Insert the PDF URL
    docs.insert([{'doc': doc_path }])

# Print all chunks and their metadata, including error columns
all_chunks = chunks_view.select(
    chunks_view.page,
    chunks_view.image,
    chunks_view.text
).collect()

print(all_chunks)

print("\n--------------------------------------")
docs.describe()
print("\n--------------------------------------")
chunks_view.describe()
print("\n--------------------------------------")

exit( 0 )