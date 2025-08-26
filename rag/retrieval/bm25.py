"""
BM25 lexical retrieval with English/Italian tokenization and metadata filters.

- Uses rank_bm25.BM25Okapi under the hood.
- Idempotent upserts keyed by our deterministic chunk IDs.
- Language-aware tokenization:
    * lowercasing
    * diacritics preserved
    * basic punctuation stripping
    * stopwords for EN/IT
- Metadata filters: equality on simple fields and tag inclusion.
- On-disk persistence (tokens+metadata+text) to ./indexes/bm25/bm25_index.jsonl by default.

NOTE: We rebuild the BM25 structure when the corpus changes (simple & robust for classroom scale).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from rag.utils.lang_detect import detect_lang_tag

# ---------------------------
# Tokenization & stopwords
# ---------------------------

# Basic latin + accented letters (Italian compatible), treat everything else as separator
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")

# Minimal but effective stopword lists (can be extended later)
_STOP_EN = {
    "a","an","the","and","or","but","if","then","else","for","to","of","in","on","at","by","with",
    "from","as","is","are","was","were","be","been","being","it","its","this","that","these","those",
    "i","you","he","she","we","they","them","his","her","their","my","your","our","me","us",
    "not","no","yes","do","does","did","doing","can","could","should","would","may","might","will","shall",
    "about","into","over","under","again","further","there","here","when","where","why","how","what","which","who","whom",
}

_STOP_IT = {
    "un","uno","una","le","la","il","lo","gli","i","l","e","o","ma","se","allora","altrimenti","per","di",
    "a","da","in","su","con","come","è","era","sono","siamo","siete","fui","fu","furono","essere","stato",
    "questo","questa","questi","queste","quello","quella","quelli","quelle","ciò","cio","io","tu","lui","lei","noi","voi","loro",
    "mio","mia","tuo","tua","suo","sua","nostro","vostro","loro","non","no","si","sia","fare","fa","fatto","posso","può","puo",
    "puoi","possono","dovrebbe","potrebbe","può","puo","sarà","sara","sarebbe","saremmo","sarete","siano","che","perché","perche",
    "quando","dove","come","cosa","quale","chi",
}

def _choose_stopwords(lang_hint: Optional[str]) -> set[str]:
    lang = (lang_hint or "").lower()
    if lang.startswith("it"):
        return _STOP_IT
    if lang.startswith("en"):
        return _STOP_EN
    # unknown → default to EN but allow detection per-doc
    return _STOP_EN

def _tokenize(text: str, lang_hint: Optional[str] = None) -> List[str]:
    """
    Tokenize to unicode words, lowercase, remove stopwords.
    If lang_hint is None, caller may pass doc-level language; otherwise we may detect at insert time.
    """
    toks = [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]
    sw = _choose_stopwords(lang_hint)
    return [t for t in toks if t not in sw and len(t) > 1]


# ---------------------------
# Filters
# ---------------------------

_FILTER_SIMPLE_FIELDS = ["course", "unit", "language", "doc_type", "author", "semester"]

def _matches_filter(meta: Mapping[str, Any], where: Optional[Mapping[str, Any]]) -> bool:
    """
    Evaluate simple 'where' filters:
      - Equality on simple fields
      - 'tags': require that all requested tags are present (AND)
      - Support {"$and":[...]} over nested simple clauses
    """
    if not where:
        return True

    # AND-list
    if "$and" in where:
        return all(_matches_filter(meta, clause) for clause in where["$and"])

    # tags $contains
    if "tags" in where and isinstance(where["tags"], dict) and "$contains" in where["tags"]:
        t = where["tags"]["$contains"]
        if not t:
            return True
        want = {t} if isinstance(t, str) else set(t)
        have = set(meta.get("tags") or [])
        return want.issubset(have)

    # simple equality
    for f in _FILTER_SIMPLE_FIELDS:
        if f in where:
            if meta.get(f) != where[f]:
                return False
    return True


# ---------------------------
# BM25 Store
# ---------------------------

@dataclass
class _Entry:
    id: str
    text: str
    tokens: List[str]
    metadata: Dict[str, Any]


@dataclass
class BM25Store:
    """
    In-memory + on-disk BM25 store.

    Persistence format: JSONL file with entries:
      {"id": "...", "text": "...", "tokens": [...], "metadata": {...}}
    We rebuild BM25Okapi from tokens on load.
    """
    index_dir: Path = Path("./indexes/bm25")
    index_file: str = "bm25_index.jsonl"

    _entries: Dict[str, _Entry] = field(default_factory=dict)     # id -> entry
    _id_list: List[str] = field(default_factory=list)             # order for BM25
    _bm25: Optional[BM25Okapi] = None

    # ---------- Core ops ----------

    def _rebuild(self) -> None:
        """Rebuild BM25 index from current entries."""
        self._id_list = list(self._entries.keys())
        corpus = [self._entries[i].tokens for i in self._id_list]
        # Avoid empty corpus crash
        self._bm25 = BM25Okapi(corpus or [[""]])

    def upsert_many(self, *, ids: Sequence[str], texts: Sequence[str], metadatas: Sequence[Mapping[str, Any]]) -> None:
        """
        Add or replace multiple documents. Tokenization uses doc metadata 'language' if present,
        otherwise falls back to detection.
        """
        if not (len(ids) == len(texts) == len(metadatas)):
            raise ValueError("ids, texts, metadatas must have the same length")

        for i, doc_id in enumerate(ids):
            text = texts[i] or ""
            meta = dict(metadatas[i] or {})
            lang = meta.get("language")
            if not lang or lang == "auto":
                # lightweight detection (en/it) if not provided
                lang = detect_lang_tag(text)
                meta["language"] = lang
            toks = _tokenize(text, lang_hint=lang)
            self._entries[doc_id] = _Entry(id=doc_id, text=text, tokens=toks, metadata=meta)

        self._rebuild()

    def delete_many(self, ids: Sequence[str]) -> None:
        for doc_id in ids:
            self._entries.pop(doc_id, None)
        self._rebuild()

    # ---------- Query ----------

    def search(self, *, query: str, where: Optional[Mapping[str, Any]] = None, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Run a BM25 search over the (optionally) filtered subset.
        Returns list of dicts with id, document, metadata, and score (higher is better).
        """
        if not query.strip() or not self._entries:
            return []

        # Evaluate filter mask
        candidate_ids = [i for i in self._id_list if _matches_filter(self._entries[i].metadata, where)]

        if not candidate_ids:
            return []

        # Build a temporary BM25 over the filtered subset (fast at classroom scale)
        corpus = [self._entries[i].tokens for i in candidate_ids]
        bm25 = BM25Okapi(corpus or [[""]])

        # Tokenize query (detect language from query text)
        q_lang = detect_lang_tag(query)
        q_tokens = _tokenize(query, lang_hint=q_lang)

        scores = bm25.get_scores(q_tokens)
        # Rank
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]

        out: List[Dict[str, Any]] = []
        for doc_id, score in ranked:
            e = self._entries[doc_id]
            out.append(
                {
                    "id": doc_id,
                    "document": e.text,
                    "metadata": e.metadata,
                    "score": float(score),
                }
            )
        return out

    # ---------- Persistence ----------

    @property
    def index_path(self) -> Path:
        return self.index_dir / self.index_file

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as f:
            for e in self._entries.values():
                rec = {
                    "id": e.id,
                    "text": e.text,
                    "tokens": e.tokens,
                    "metadata": e.metadata,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load(self) -> None:
        self._entries.clear()
        if not self.index_path.exists():
            self._rebuild()
            return
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self._entries[rec["id"]] = _Entry(
                    id=rec["id"],
                    text=rec.get("text", ""),
                    tokens=list(rec.get("tokens", [])),
                    metadata=dict(rec.get("metadata", {})),
                )
        self._rebuild()

    # ---------- Convenience ----------

    @classmethod
    def load_or_create(cls, index_dir: str | Path = "./indexes/bm25") -> "BM25Store":
        store = cls(index_dir=Path(index_dir))
        store.load()
        return store
