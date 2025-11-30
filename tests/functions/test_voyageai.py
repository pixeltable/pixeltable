import pytest

import pixeltable as pxt

from ..utils import get_image_files, rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestVoyageAI:
    @pytest.mark.parametrize('output_dimension', [None, 512])
    def test_embeddings(self, reset_db: None, output_dimension: int | None) -> None:
        skip_test_if_not_installed('voyageai')
        skip_test_if_no_client('voyageai')
        from pixeltable.functions.voyageai import embeddings

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        if output_dimension is None:
            t.add_computed_column(embed=embeddings(t.input, model='voyage-3.5', input_type='document'))
            expected_dim = 1024  # default dimension for voyage-3.5
        else:
            t.add_computed_column(
                embed=embeddings(t.input, model='voyage-3.5', input_type='document', output_dimension=output_dimension)
            )
            expected_dim = output_dimension

        validate_update_status(t.insert(input='Voyage AI provides cutting-edge embeddings and rerankers.'), 1)
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (expected_dim,)

    def test_embeddings_index(self, reset_db: None) -> None:
        """Test using Voyage AI embeddings with an embedding index."""
        skip_test_if_not_installed('voyageai')
        skip_test_if_no_client('voyageai')
        from pixeltable.functions.voyageai import embeddings

        # Create a simple table with text
        t = pxt.create_table('docs', {'text': pxt.String})

        # Create embedding function for indexing
        embed_fn = embeddings.using(model='voyage-3.5', input_type='document')

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
        sim = t.text.similarity('What is machine learning?')
        results = t.order_by(sim, asc=False).limit(2).select(t.text, similarity=sim).collect()

        assert len(results) == 2
        # The ML-related text should be ranked first
        assert (
            'machine learning' in results['text'][0].lower() or 'artificial intelligence' in results['text'][0].lower()
        )

    def test_rerank(self, reset_db: None) -> None:
        skip_test_if_not_installed('voyageai')
        skip_test_if_no_client('voyageai')
        from pixeltable.functions.voyageai import rerank

        documents = [
            'The Mediterranean diet emphasizes fish, olive oil, and vegetables.',
            'Photosynthesis in plants converts light energy into glucose.',
            '20th-century innovations, from radios to smartphones.',
            'Rivers provide water, irrigation, and habitat for aquatic species.',
            "Apple's conference call is scheduled for Thursday, November 2, 2023.",
            "Shakespeare's works, like 'Hamlet', endure in literature.",
        ]

        t = pxt.create_table('test_tbl', {'query': pxt.String, 'docs': pxt.Json})
        t.add_computed_column(reranked=rerank(t.query, t.docs, model='rerank-2.5-lite', top_k=3))

        validate_update_status(t.insert(query="When is Apple's conference call scheduled?", docs=documents), 1)

        res = t.select(t.reranked).collect()
        result = res['reranked'][0]

        # Verify structure
        assert 'results' in result
        assert 'total_tokens' in result
        assert len(result['results']) == 3  # We asked for top_k=3

        # Verify first result has highest relevance and is about Apple
        first_result = result['results'][0]
        assert 'index' in first_result
        assert 'document' in first_result
        assert 'relevance_score' in first_result
        assert 'Apple' in first_result['document']

        # Verify scores are in descending order
        scores = [r['relevance_score'] for r in result['results']]
        assert scores == sorted(scores, reverse=True)

    def test_multimodal_embed(self, reset_db: None) -> None:
        """Test multimodal embeddings with images."""
        skip_test_if_not_installed('voyageai')
        skip_test_if_no_client('voyageai')
        from pixeltable.functions.voyageai import multimodal_embed

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t.add_computed_column(embed=multimodal_embed(t.img, input_type='document'))

        # Use a test image
        img_paths = get_image_files()
        validate_update_status(t.insert(img=img_paths[0]), 1)
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (1024,)  # voyage-multimodal-3 produces 1024-dim embeddings
