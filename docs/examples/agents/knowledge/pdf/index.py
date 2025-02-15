import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import DocumentSplitter

def create_index(index_name: str, chunks_name: str, reset_history: bool = False) -> tuple[pxt.Table, pxt.Table]:
    # Delete index if it exists
    if reset_history:
        pxt.drop_table(index_name, force=True)

    # Create index if it doesn't exist
    if index_name not in pxt.list_tables():

        document_table = pxt.create_table(index_name, {'pdf': pxt.Document}, if_exists='ignore')

        # Create view that chunks PDFs into sections
        chunks_view = pxt.create_view(
            chunks_name,
            document_table,
            iterator=DocumentSplitter.create(document=document_table.pdf, separators='token_limit', limit=300),
            if_exists='ignore',
        )

        # Define the embedding model
        embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

        # Create embedding index
        chunks_view.add_embedding_index(column='text', string_embed=embed_model)