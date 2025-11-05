from nerxiv.utils.caching import compute_chunker_hash, compute_retriever_hash


class TestComputeChunkerHash:
    """Tests for the compute_chunker_hash function."""

    def test_hash_is_consistent(self):
        """Tests that the same inputs produce the same hash."""
        text = "This is a test text."
        chunker_name = "Chunker"
        chunker_params = {"chunk_size": 1000, "chunk_overlap": 200}

        hash1 = compute_chunker_hash(text, chunker_name, chunker_params)
        hash2 = compute_chunker_hash(text, chunker_name, chunker_params)

        assert hash1 == hash2

    def test_different_text_produces_different_hash(self):
        """Tests that different texts produce different hashes."""
        chunker_name = "Chunker"
        chunker_params = {"chunk_size": 1000, "chunk_overlap": 200}

        hash1 = compute_chunker_hash("Text A", chunker_name, chunker_params)
        hash2 = compute_chunker_hash("Text B", chunker_name, chunker_params)

        assert hash1 != hash2

    def test_different_chunker_produces_different_hash(self):
        """Tests that different chunkers produce different hashes."""
        text = "This is a test text."
        chunker_params = {}

        hash1 = compute_chunker_hash(text, "Chunker", chunker_params)
        hash2 = compute_chunker_hash(text, "SemanticChunker", chunker_params)

        assert hash1 != hash2

    def test_different_params_produce_different_hash(self):
        """Tests that different parameters produce different hashes."""
        text = "This is a test text."
        chunker_name = "Chunker"

        hash1 = compute_chunker_hash(
            text, chunker_name, {"chunk_size": 1000, "chunk_overlap": 200}
        )
        hash2 = compute_chunker_hash(
            text, chunker_name, {"chunk_size": 500, "chunk_overlap": 100}
        )

        assert hash1 != hash2

    def test_hash_with_no_params(self):
        """Tests that hash works with no parameters (SemanticChunker case)."""
        text = "This is a test text."
        chunker_name = "SemanticChunker"

        hash1 = compute_chunker_hash(text, chunker_name, None)
        hash2 = compute_chunker_hash(text, chunker_name, {})

        # None and {} should produce the same hash
        assert hash1 == hash2

    def test_hash_format(self):
        """Tests that the hash is in the expected format (SHA256 hex)."""
        text = "This is a test text."
        chunker_name = "Chunker"
        chunker_params = {"chunk_size": 1000}

        hash_value = compute_chunker_hash(text, chunker_name, chunker_params)

        # SHA256 produces a 64-character hexadecimal string
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestComputeRetrieverHash:
    """Tests for the compute_retriever_hash function."""

    def test_hash_is_consistent(self):
        """Tests that the same inputs produce the same hash."""
        chunker_hash = "abc123"
        retriever_params = {
            "retriever_model": "all-MiniLM-L6-v2",
            "retriever_query": "test query",
            "n_top_chunks": 5,
        }

        hash1 = compute_retriever_hash(chunker_hash, retriever_params)
        hash2 = compute_retriever_hash(chunker_hash, retriever_params)

        assert hash1 == hash2

    def test_different_chunker_hash_produces_different_hash(self):
        """Tests that different chunker hashes produce different retriever hashes."""
        retriever_params = {
            "retriever_model": "all-MiniLM-L6-v2",
            "retriever_query": "test query",
            "n_top_chunks": 5,
        }

        hash1 = compute_retriever_hash("chunker_hash1", retriever_params)
        hash2 = compute_retriever_hash("chunker_hash2", retriever_params)

        assert hash1 != hash2

    def test_different_retriever_model_produces_different_hash(self):
        """Tests that different retriever models produce different hashes."""
        chunker_hash = "abc123"
        retriever_params = {
            "retriever_query": "test query",
            "n_top_chunks": 5,
        }

        hash1 = compute_retriever_hash(
            chunker_hash, {"retriever_model": "model1", **retriever_params}
        )
        hash2 = compute_retriever_hash(
            chunker_hash, {"retriever_model": "model2", **retriever_params}
        )

        assert hash1 != hash2

    def test_different_query_produces_different_hash(self):
        """Tests that different queries produce different hashes."""
        chunker_hash = "abc123"
        retriever_params = {
            "retriever_model": "all-MiniLM-L6-v2",
            "n_top_chunks": 5,
        }

        hash1 = compute_retriever_hash(
            chunker_hash, {"retriever_query": "query1", **retriever_params}
        )
        hash2 = compute_retriever_hash(
            chunker_hash, {"retriever_query": "query2", **retriever_params}
        )

        assert hash1 != hash2

    def test_different_n_top_chunks_produces_different_hash(self):
        """Tests that different n_top_chunks values produce different hashes."""
        chunker_hash = "abc123"
        retriever_params = {
            "retriever_model": "all-MiniLM-L6-v2",
            "retriever_query": "test query",
        }

        hash1 = compute_retriever_hash(
            chunker_hash, {"n_top_chunks": 5, **retriever_params}
        )
        hash2 = compute_retriever_hash(
            chunker_hash, {"n_top_chunks": 10, **retriever_params}
        )

        assert hash1 != hash2

    def test_hash_format(self):
        """Tests that the hash is in the expected format (SHA256 hex)."""
        chunker_hash = "abc123"
        retriever_params = {
            "retriever_model": "all-MiniLM-L6-v2",
            "retriever_query": "test query",
            "n_top_chunks": 5,
        }

        hash_value = compute_retriever_hash(chunker_hash, retriever_params)

        # SHA256 produces a 64-character hexadecimal string
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)
