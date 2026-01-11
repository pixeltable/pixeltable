import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestJina:
    def test_embeddings(self, reset_db: None) -> None:
        """Test basic embedding generation."""
        skip_test_if_no_client('jina')
        from pixeltable.functions.jina import embeddings

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(embed=embeddings(t.input, model='jina-embeddings-v3', task='retrieval.passage'))

        validate_update_status(t.insert(input='Jina AI provides cutting-edge embeddings and rerankers.'), 1)
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (1024,)  # jina-embeddings-v3 produces 1024-dim embeddings

    def test_embeddings_with_dimensions(self, reset_db: None) -> None:
        """Test embedding generation with custom dimensions."""
        skip_test_if_no_client('jina')
        from pixeltable.functions.jina import embeddings

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        # jina-embeddings-v3 supports custom dimensions
        t.add_computed_column(
            embed=embeddings(t.input, model='jina-embeddings-v3', task='retrieval.passage', dimensions=512)
        )

        validate_update_status(t.insert(input='Testing custom embedding dimensions.'), 1)
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (512,)

    def test_embeddings_index(self, reset_db: None) -> None:
        """Test using Jina embeddings with an embedding index."""
        skip_test_if_no_client('jina')
        from pixeltable.functions.jina import embeddings

        t = pxt.create_table('docs', {'text': pxt.String})

        # Create embedding function for indexing
        embed_fn = embeddings.using(model='jina-embeddings-v3', task='retrieval.passage')

        # Add embedding index
        t.add_embedding_index('text', string_embed=embed_fn)

        # Insert some documents
        validate_update_status(
            t.insert(
                [
                    {'text': 'Machine learning is a subset of artificial intelligence.'},
                    {'text': 'Deep learning uses neural networks with many layers.'},
                    {'text': 'Python is a popular programming language.'},
                ]
            ),
            3,
        )

        # Test similarity search using the index
        sim = t.text.similarity(string='What is machine learning?')
        results = t.order_by(sim, asc=False).limit(2).select(t.text, similarity=sim).collect()

        assert len(results) == 2
        # The ML-related text should be ranked first
        assert 'machine learning' in results['text'][0].lower()

    def test_rerank(self, reset_db: None) -> None:
        """Test document reranking."""
        skip_test_if_no_client('jina')
        from pixeltable.functions.jina import rerank

        documents = [
            'The Mediterranean diet emphasizes fish, olive oil, and vegetables.',
            'Photosynthesis in plants converts light energy into glucose.',
            '20th-century innovations, from radios to smartphones.',
            'Rivers provide water, irrigation, and habitat for aquatic species.',
            "Apple's conference call is scheduled for Thursday, November 2, 2023.",
            "Shakespeare's works, like 'Hamlet', endure in literature.",
        ]

        t = pxt.create_table('test_tbl', {'query': pxt.String, 'docs': pxt.Json})
        t.add_computed_column(
            reranked=rerank(t.query, t.docs, model='jina-reranker-v2-base-multilingual', top_n=3, return_documents=True)
        )

        validate_update_status(t.insert(query="When is Apple's conference call scheduled?", docs=documents), 1)

        res = t.select(t.reranked).collect()
        result = res['reranked'][0]

        # Verify structure
        assert 'results' in result
        assert 'usage' in result
        assert len(result['results']) == 3  # We asked for top_n=3

        # Verify first result has highest relevance and is about Apple
        first_result = result['results'][0]
        assert 'index' in first_result
        assert 'relevance_score' in first_result
        assert 'document' in first_result  # We set return_documents=True
        assert 'Apple' in first_result['document']

        # Verify scores are in descending order
        scores = [r['relevance_score'] for r in result['results']]
        assert scores == sorted(scores, reverse=True)
