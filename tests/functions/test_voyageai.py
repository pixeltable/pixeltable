import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestVoyageAI:
    def test_embeddings(self, reset_db: None) -> None:
        skip_test_if_not_installed('voyageai')
        skip_test_if_no_client('voyageai')
        from pixeltable.functions.voyageai import embeddings

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(embed=embeddings(t.input, model='voyage-3.5', input_type='document'))
        validate_update_status(
            t.insert(input='Voyage AI provides cutting-edge embeddings and rerankers.'),
            1
        )
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (1024,)

    def test_embeddings_with_dimensions(self, reset_db: None) -> None:
        skip_test_if_not_installed('voyageai')
        skip_test_if_no_client('voyageai')
        from pixeltable.functions.voyageai import embeddings

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(
            embed=embeddings(t.input, model='voyage-3.5', input_type='document', output_dimension=512)
        )
        validate_update_status(
            t.insert(input='Voyage AI provides cutting-edge embeddings and rerankers.'),
            1
        )
        res = t.select(t.embed).collect()
        assert res['embed'][0].shape == (512,)

    def test_rerank(self, reset_db: None) -> None:
        skip_test_if_not_installed('voyageai')
        skip_test_if_no_client('voyageai')
        from pixeltable.functions.voyageai import rerank

        documents = [
            "The Mediterranean diet emphasizes fish, olive oil, and vegetables.",
            "Photosynthesis in plants converts light energy into glucose.",
            "20th-century innovations, from radios to smartphones.",
            "Rivers provide water, irrigation, and habitat for aquatic species.",
            "Apple's conference call is scheduled for Thursday, November 2, 2023.",
            "Shakespeare's works, like 'Hamlet', endure in literature."
        ]

        t = pxt.create_table('test_tbl', {'query': pxt.String, 'docs': pxt.Json})
        t.add_computed_column(
            reranked=rerank(t.query, t.docs, model='rerank-2.5-lite', top_k=3)
        )
        
        validate_update_status(
            t.insert(query="When is Apple's conference call scheduled?", docs=documents),
            1
        )
        
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

