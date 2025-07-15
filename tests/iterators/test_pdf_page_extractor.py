import io
import os
from typing import List

from PIL import Image

from pixeltable.iterators import PdfPageExtractor

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
print( "\nINGEST the following files: " , pdfs, "\n" )

for doc_path in pdfs:

    extractor = PdfPageExtractor(document=doc_path)

    for chunk in extractor:
        print(f'Page: {chunk["page"]}')
        print(f'Text preview: {chunk["text"][:100]}')
        width, height = chunk["image"].size
        print(f'Image size: {width}x{height} pixels\n')
        # print(f'Image bytes: {len(chunk["image"])} bytes\n')

        img_io = io.BytesIO()
        chunk["image"].save(img_io, format="PNG")
        img_bytes = img_io.getvalue()

        print(f"Image size in bytes: {len(img_bytes)}")

        img = Image.open(io.BytesIO(img_bytes))
        img.show()

    extractor.close()