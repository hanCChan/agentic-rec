"""Import Rec-R1 Python modules by file path (avoid `src` package clash)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _agentic_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_pyserini_search_class(rec_r1_root: Path | None = None) -> Any:
    rec_r1_root = Path(rec_r1_root or _agentic_root() / "Rec-R1")
    search_path = rec_r1_root / "src/Lucene/esci/search.py"
    mod = load_module_from_path("rec_r1_lucene_search", search_path)
    return mod.PyseriniMultiFieldSearch


def get_ndcg_at_k(rec_r1_root: Path | None = None) -> Callable[..., float]:
    rec_r1_root = Path(rec_r1_root or _agentic_root() / "Rec-R1")
    utils_path = rec_r1_root / "src/Lucene/utils.py"
    mod = load_module_from_path("rec_r1_lucene_utils", utils_path)
    return mod.ndcg_at_k
