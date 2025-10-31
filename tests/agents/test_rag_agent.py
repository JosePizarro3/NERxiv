from nerxiv.agents.rag_agent import RAGExtractorAgent


class SimpleChunker:
    def __init__(self, text: str = "", **kwargs):
        if not text:
            raise ValueError("text required")
        self.text = text

    def chunk_text(self):
        # naive split by sentences
        return [
            type("D", (), {"page_content": s})()
            for s in self.text.split(".")
            if s.strip()
        ]


class SimpleRetriever:
    def __init__(self, query: str = "", **kwargs):
        if query is None:
            raise ValueError("query required")
        self.query = query

    def get_relevant_chunks(self, chunks, n_top_chunks=5):
        # return first n chunks joined
        items = [c.page_content for c in chunks][:n_top_chunks]
        return "\n\n".join(items)


class SimpleGenerator:
    def __init__(self, text: str = "", **kwargs):
        if text is None:
            raise ValueError("text required")
        self.text = text

    def generate(self, prompt: str = "", regex: str = "", del_regex: str = ""):
        return f"GENERATED:\nContext:\n{self.text}\nPrompt:\n{prompt}"


def test_rag_agent_smoke_flow():
    text = "This is sentence one. This is sentence two. This is sentence three."
    query = "What is sentence two about?"

    agent = RAGExtractorAgent(
        chunker=SimpleChunker,
        retriever=SimpleRetriever,
        generator=SimpleGenerator,
    )

    result = agent.extract(text=text, query=query, n_top_chunks=2)

    assert "chunks" in result
    assert "retrieved" in result
    assert "prompt" in result
    assert "answer" in result
    assert "GENERATED:" in result["answer"]
