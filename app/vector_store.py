import re
from typing import Any, Dict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self, persist_directory="data/chroma"):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="papers"
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def _get_id_candidates(self, paper_id: str) -> List[str]:
        paper_id = paper_id.strip()
        clean_id = re.sub(r"v\d+$", "", paper_id)
        candidates = [paper_id, clean_id]
        for v in range(1, 11):
            candidates.append(f"{clean_id}v{v}")
        return list(dict.fromkeys(candidates))

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        documents = [chunk["text"] for chunk in chunks]
        ids = [f'{chunk["paper_id"]}_{chunk["chunk_id"]}' for chunk in chunks]

        metadatas = [
            {
                "paper_id": chunk["paper_id"],
                "chunk_id": chunk["chunk_id"],
                "pages": str(chunk.get("pages", [])),
                "start_page": chunk.get("start_page", chunk["pages"][0] if chunk.get("pages") else 1),
                "end_page": chunk.get("end_page", chunk["pages"][-1] if chunk.get("pages") else 1),
            }
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=False
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query: str, paper_id: Optional[str] = None, top_k: int = 5):
        query_embedding = self.embedding_model.encode(
            [query],
            show_progress_bar=False
        ).tolist()

        query_kwargs = {
            "query_embeddings": query_embedding,
            "n_results": top_k
        }
        if paper_id:
            candidates = self._get_id_candidates(paper_id)
            query_kwargs["where"] = {"paper_id": {"$in": candidates}}

        results = self.collection.query(**query_kwargs)
        return results

    def has_paper(self, paper_id: str) -> bool:
        try:
            candidates = self._get_id_candidates(paper_id)
            res = self.collection.get(where={"paper_id": {"$in": candidates}}, limit=1)
            return bool(res and res.get("ids"))
        except Exception:
            return False