"""
core/rag.py — LlamaIndex RAG pipeline over local codebase (Cluster 03: Sovereign Developer OS)

Indexes a local directory of code files into Chroma via Ollama embeddings.
Exposes a query function that retrieves relevant context and generates
a response using the local LLM.

SDKs: LlamaIndex, Chroma, Ollama (nomic-embed-text + llama3)
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CodebaseRAG:
    """
    RAG pipeline for local codebases.
    - Ingests files from workspace paths
    - Embeds with Ollama nomic-embed-text
    - Stores in Chroma (persisted locally)
    - Queries with hybrid retrieval + Ollama LLM
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._rag_cfg = config.get("rag", {})
        self._ollama_cfg = config.get("ollama", {})
        self._chroma_cfg = config.get("chroma", {})

        self._chunk_size    = self._rag_cfg.get("chunk_size", 512)
        self._chunk_overlap = self._rag_cfg.get("chunk_overlap", 64)
        self._top_k         = self._rag_cfg.get("top_k", 8)
        self._base_url      = self._ollama_cfg.get("base_url", "http://localhost:11434")
        self._llm_model     = self._ollama_cfg.get("model", "llama3")
        self._embed_model   = self._ollama_cfg.get("embed_model", "nomic-embed-text")
        self._persist_dir   = os.path.expanduser(self._chroma_cfg.get("persist_dir", "~/.sovereign/chroma"))
        self._collection    = self._chroma_cfg.get("collection", "codebase")

        self._index = None
        self._query_engine = None

    def _build_llm(self):
        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=self._llm_model,
            base_url=self._base_url,
            request_timeout=120.0,
            temperature=self._cfg.get("agent", {}).get("temperature", 0.2),
        )

    def _build_embed(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        return OllamaEmbedding(
            model_name=self._embed_model,
            base_url=self._base_url,
        )

    def _build_chroma_store(self):
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        os.makedirs(self._persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=self._persist_dir)
        collection = client.get_or_create_collection(self._collection)
        return ChromaVectorStore(chroma_collection=collection)

    def _get_extensions(self) -> list:
        return self._cfg.get("workspace", {}).get(
            "extensions", [".py", ".ts", ".js", ".go", ".rs", ".md", ".json"]
        )

    def index(self, paths: list, force_reindex: bool = False) -> int:
        """
        Index files from the given paths into Chroma.
        Returns number of documents indexed.
        """
        from llama_index.core import (
            VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
        )
        from llama_index.core.node_parser import CodeSplitter, SentenceSplitter

        Settings.llm = self._build_llm()
        Settings.embed_model = self._build_embed()

        vector_store = self._build_chroma_store()
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

        all_docs = []
        extensions = self._get_extensions()

        for path in paths:
            expanded = os.path.expanduser(path)
            if not os.path.exists(expanded):
                logger.warning("Path does not exist: %s", expanded)
                continue

            logger.info("Loading documents from: %s", expanded)
            try:
                reader = SimpleDirectoryReader(
                    input_dir=expanded,
                    recursive=True,
                    required_exts=extensions,
                    exclude_hidden=True,
                )
                docs = reader.load_data()
                all_docs.extend(docs)
                logger.info("Loaded %d docs from %s", len(docs), expanded)
            except Exception as e:
                logger.error("Failed to load from %s: %s", expanded, e)

        if not all_docs:
            logger.warning("No documents found to index")
            return 0

        # Use CodeSplitter for code files, SentenceSplitter for text/markdown
        splitter = SentenceSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        logger.info("Indexing %d documents into Chroma...", len(all_docs))
        self._index = VectorStoreIndex.from_documents(
            all_docs,
            storage_context=storage_ctx,
            transformations=[splitter],
            show_progress=True,
        )
        self._query_engine = self._index.as_query_engine(
            similarity_top_k=self._top_k,
            streaming=False,
        )
        logger.info("Indexing complete. %d documents indexed.", len(all_docs))
        return len(all_docs)

    def load_existing(self) -> bool:
        """
        Load an existing Chroma index without re-indexing.
        Returns True if index was found and loaded.
        """
        from llama_index.core import VectorStoreIndex, StorageContext, Settings

        Settings.llm = self._build_llm()
        Settings.embed_model = self._build_embed()

        try:
            vector_store = self._build_chroma_store()
            storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_ctx,
            )
            self._query_engine = self._index.as_query_engine(
                similarity_top_k=self._top_k,
            )
            logger.info("Loaded existing Chroma index from %s", self._persist_dir)
            return True
        except Exception as e:
            logger.warning("Could not load existing index: %s", e)
            return False

    def query(self, question: str, context: Optional[str] = None) -> dict:
        """
        Query the indexed codebase.
        Returns {answer, source_files, node_count}.
        """
        if self._query_engine is None:
            loaded = self.load_existing()
            if not loaded:
                return {"answer": "No index found. Run index() first.", "source_files": [], "node_count": 0}

        full_question = question
        if context:
            full_question = f"Context: {context}\n\nQuestion: {question}"

        try:
            response = self._query_engine.query(full_question)
            source_files = list({
                node.metadata.get("file_path", node.metadata.get("file_name", "unknown"))
                for node in response.source_nodes
            })
            return {
                "answer": str(response),
                "source_files": source_files,
                "node_count": len(response.source_nodes),
            }
        except Exception as e:
            logger.error("Query failed: %s", e)
            return {"answer": f"Query error: {e}", "source_files": [], "node_count": 0}

    def query_stream(self, question: str):
        """Streaming query — yields token strings."""
        if self._query_engine is None:
            self.load_existing()

        streaming_engine = self._index.as_query_engine(
            similarity_top_k=self._top_k,
            streaming=True,
        )
        try:
            response = streaming_engine.query(question)
            for token in response.response_gen:
                yield token
        except Exception as e:
            yield f"[stream error: {e}]"

    def get_stats(self) -> dict:
        """Return index statistics."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self._persist_dir)
            collection = client.get_or_create_collection(self._collection)
            count = collection.count()
            return {
                "collection": self._collection,
                "persist_dir": self._persist_dir,
                "chunk_count": count,
                "llm_model": self._llm_model,
                "embed_model": self._embed_model,
                "top_k": self._top_k,
                "index_loaded": self._index is not None,
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Codebase RAG — index and query local code")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--action", choices=["index", "query", "stats"], default="stats")
    parser.add_argument("--path", nargs="+", help="Paths to index (overrides config)")
    parser.add_argument("--question", help="Question to ask")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    rag = CodebaseRAG(cfg)

    if args.action == "index":
        paths = args.path or cfg.get("workspace", {}).get("paths", ["."])
        n = rag.index(paths)
        print(f"Indexed {n} documents")

    elif args.action == "query":
        if not args.question:
            print("--question required for query action")
        else:
            result = rag.query(args.question)
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources ({result['node_count']} nodes):")
            for f in result["source_files"]:
                print(f"  {f}")

    elif args.action == "stats":
        print(json.dumps(rag.get_stats(), indent=2))
