"""BM25 search tool wrapper around Rec-R1 Pyserini index."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple


class BM25SearchTool:
    def __init__(
        self,
        index_dir: str | Path | None = None,
        rec_r1_root: str | Path | None = None,
        threads: int = 16,
    ):
        rec_r1_root = Path(rec_r1_root or Path(__file__).resolve().parents[2] / "Rec-R1")
        from src.rec_r1_bridge import get_pyserini_search_class

        PyseriniMultiFieldSearch = get_pyserini_search_class(rec_r1_root)

        index_dir = Path(index_dir or rec_r1_root / "database/esci/pyserini_index")
        if not index_dir.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_dir}")

        self.index_dir = index_dir
        self.rec_r1_root = rec_r1_root
        self.threads = threads
        self._searcher = PyseriniMultiFieldSearch(index_dir=str(index_dir))

    def search(self, query: str, topk: int = 20) -> List[Tuple[str, str, float]]:
        results = self._searcher.batch_search([query], top_k=topk, threads=self.threads)
        return results.get(query, [])

    def format_observation(self, query: str, hits: Sequence[Tuple[str, str, float]], max_items: int = 5) -> str:
        lines = [f"query={query!r} hits={len(hits)}"]
        for rank, (doc_id, content, score) in enumerate(hits[:max_items], start=1):
            snippet = " ".join(content.split())[:160]
            lines.append(f"top{rank}: id={doc_id} score={score:.3f} text={snippet}")
        if len(hits) > max_items:
            lines.append(f"... {len(hits) - max_items} more results omitted")
        return "\n".join(lines)

    def retrieved_ids(self, query: str, topk: int = 20) -> List[str]:
        return [doc_id for doc_id, _, _ in self.search(query, topk=topk)]
