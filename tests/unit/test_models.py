"""Model validation tests for API request schemas."""

import pytest
from pydantic import ValidationError

from seal_embedding_api.models import SimilaritySearchRequest


def test_similarity_search_requires_query():
    """Require either image_b64 or query_seal_id."""
    with pytest.raises(ValidationError):
        SimilaritySearchRequest()


def test_similarity_search_top_k_min():
    """Reject top_k < 1."""
    with pytest.raises(ValidationError):
        SimilaritySearchRequest(query_seal_id="uid_1", top_k=0)
