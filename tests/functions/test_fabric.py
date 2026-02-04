"""
Tests for Microsoft Fabric integration.

These tests only run in Microsoft Fabric notebook environments where the
synapse-ml-fabric package is available and Fabric authentication is configured.
"""

import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_not_installed, validate_update_status


def _skip_if_not_fabric() -> None:
    """Skip test if not running in Fabric environment."""
    skip_test_if_not_installed("synapse.ml.fabric")
    # Try to get fabric config - will raise if not in Fabric
    try:
        from synapse.ml.fabric.service_discovery import get_fabric_env_config

        _ = get_fabric_env_config()
    except Exception as e:
        pytest.skip(f"Not running in Fabric environment: {e}")


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestFabric:
    """Test suite for Microsoft Fabric Azure OpenAI integration."""

    def test_chat_completions_standard_model(self, uses_db: None) -> None:
        """Test chat completions with a standard model (gpt-4.1)."""
        _skip_if_not_fabric()
        from pixeltable.functions import fabric

        t = pxt.create_table("test_tbl", {"input": pxt.String})
        messages = [{"role": "user", "content": t.input}]
        t.add_computed_column(
            output=fabric.chat_completions(
                messages,
                model="gpt-4.1",
                model_kwargs={"max_tokens": 100, "temperature": 0.7},
            )
        )
        validate_update_status(t.insert(input="What is 2+2?"), 1)
        results = t.collect()
        assert len(results["output"]) > 0
        assert "choices" in results["output"][0]
        assert len(results["output"][0]["choices"]) > 0

    def test_chat_completions_reasoning_model(self, uses_db: None) -> None:
        """Test chat completions with a reasoning model (gpt-5)."""
        _skip_if_not_fabric()
        from pixeltable.functions import fabric

        t = pxt.create_table("test_tbl", {"input": pxt.String})
        messages = [{"role": "user", "content": t.input}]
        # Reasoning models use max_completion_tokens, not max_tokens
        t.add_computed_column(
            output=fabric.chat_completions(
                messages, model="gpt-5", model_kwargs={"max_completion_tokens": 100}
            )
        )
        validate_update_status(t.insert(input="Explain the concept of recursion."), 1)
        results = t.collect()
        assert len(results["output"]) > 0
        assert "choices" in results["output"][0]
        assert len(results["output"][0]["choices"]) > 0

    def test_chat_completions_with_max_tokens_conversion(self, uses_db: None) -> None:
        """Test that max_tokens is converted to max_completion_tokens for reasoning models."""
        _skip_if_not_fabric()
        from pixeltable.functions import fabric

        t = pxt.create_table("test_tbl", {"input": pxt.String})
        messages = [{"role": "user", "content": t.input}]
        # Test that max_tokens in model_kwargs is automatically converted for reasoning models
        t.add_computed_column(
            output=fabric.chat_completions(
                messages,
                model="gpt-5",
                model_kwargs={
                    "max_tokens": 100
                },  # Should be converted to max_completion_tokens
            )
        )
        validate_update_status(t.insert(input="What is AI?"), 1)
        results = t.collect()
        assert len(results["output"]) > 0
        assert "choices" in results["output"][0]

    def test_embeddings(self, uses_db: None) -> None:
        """Test embeddings generation."""
        _skip_if_not_fabric()
        from pixeltable.functions import fabric

        t = pxt.create_table("test_tbl", {"text": pxt.String})
        t.add_computed_column(
            embed=fabric.embeddings(t.text, model="text-embedding-ada-002")
        )
        validate_update_status(
            t.insert(
                [
                    {"text": "Hello, world!"},
                    {"text": "Pixeltable is great for AI workflows."},
                ]
            ),
            2,
        )
        results = t.collect()
        assert len(results["embed"]) == 2
        # Check that embeddings are arrays
        assert all(hasattr(emb, "__len__") for emb in results["embed"])
        # Check that all embeddings have the same length
        assert len(set(len(emb) for emb in results["embed"])) == 1
        # text-embedding-ada-002 produces 1536-dimensional vectors
        assert len(results["embed"][0]) == 1536

    def test_embeddings_batching(self, uses_db: None) -> None:
        """Test that embeddings handle batching correctly."""
        _skip_if_not_fabric()
        from pixeltable.functions import fabric

        t = pxt.create_table("test_tbl", {"text": pxt.String})
        t.add_computed_column(embed=fabric.embeddings(t.text))

        # Insert multiple rows to test batching (batch_size=32)
        texts = [{"text": f"Sample text number {i}"} for i in range(50)]
        validate_update_status(t.insert(texts), 50)

        results = t.collect()
        assert len(results["embed"]) == 50
        # All embeddings should be valid arrays
        assert all(hasattr(emb, "__len__") for emb in results["embed"])

    def test_chat_completions_custom_api_version(self, uses_db: None) -> None:
        """Test chat completions with custom API version."""
        _skip_if_not_fabric()
        from pixeltable.functions import fabric

        t = pxt.create_table("test_tbl", {"input": pxt.String})
        messages = [{"role": "user", "content": t.input}]
        # Test with explicit API version
        t.add_computed_column(
            output=fabric.chat_completions(
                messages, model="gpt-4.1", api_version="2024-02-15-preview"
            )
        )
        validate_update_status(t.insert(input="Hello"), 1)
        results = t.collect()
        assert len(results["output"]) > 0
        assert "choices" in results["output"][0]

    def test_embeddings_custom_api_version(self, uses_db: None) -> None:
        """Test embeddings with custom API version."""
        _skip_if_not_fabric()
        from pixeltable.functions import fabric

        t = pxt.create_table("test_tbl", {"text": pxt.String})
        t.add_computed_column(
            embed=fabric.embeddings(
                t.text, model="text-embedding-ada-002", api_version="2024-02-15-preview"
            )
        )
        validate_update_status(t.insert(text="Test text"), 1)
        results = t.collect()
        assert len(results["embed"]) == 1
        assert len(results["embed"][0]) == 1536
