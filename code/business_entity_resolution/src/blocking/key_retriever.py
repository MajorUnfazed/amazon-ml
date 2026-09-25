"""Rule- and key-based candidate generation blocks for LinkSure."""

from typing import List, Dict, Tuple, Set

class KeyRetriever:
    """
    Inverted-index key blocker for exact postal code and acronym matches.
    """
    def __init__(self, name: str):
        self.name = name
        self.index: Dict[str, List[str]] = {}

    def fit_index(self, keys: List[str], partner_ids: List[str]):
        """Builds key -> list of partner IDs index."""
        self.index.clear()
        for k, pid in zip(keys, partner_ids):
            k_clean = str(k).strip().lower()
            if k_clean and len(k_clean) >= 3:
                self.index.setdefault(k_clean, []).append(pid)

    def retrieve(
        self,
        query_keys: List[str],
        query_ids: List[str],
        max_candidates_per_key: int = 30
    ) -> List[Tuple[str, str, float, str]]:
        """
        Retrieves candidates sharing identical non-empty keys.
        """
        results = []
        for q_id, q_key in zip(query_ids, query_keys):
            k_clean = str(q_key).strip().lower()
            if not k_clean or len(k_clean) < 3:
                continue

            matches = self.index.get(k_clean, [])
            # If a key is too broad (e.g. city name with 10k entities), cap matches
            for pid in matches[:max_candidates_per_key]:
                results.append((q_id, pid, 1.0, self.name))

        return results
