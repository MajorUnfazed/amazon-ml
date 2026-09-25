"""Multi-retriever union and candidate management for LinkSure."""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from .tfidf_retriever import TfidfRetriever
from .key_retriever import KeyRetriever

class CandidateManager:
    """
    Coordinates multi-retriever candidate generation:
    1. Name Core Char TF-IDF
    2. Name + Address Char TF-IDF
    3. Address Word TF-IDF
    4. Exact Postal Code Block
    5. Acronym Key Block
    """
    def __init__(
        self,
        top_k_name: int = 25,
        top_k_name_addr: int = 15,
        top_k_addr: int = 15,
        max_candidates_per_s1: int = 50,
        scope_by_country: bool = True
    ):
        self.top_k_name = top_k_name
        self.top_k_name_addr = top_k_name_addr
        self.top_k_addr = top_k_addr
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.scope_by_country = scope_by_country

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        partner_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Runs all retrievers and returns DataFrame of candidate pairs:
        Columns: [source1_entity_id, partner_entity_id, score_name_tfidf, score_name_addr_tfidf, score_addr_tfidf, score_postal_block, score_acronym_block, max_retriever_score]
        """
        if self.scope_by_country and "country_clean" in s1_df.columns and "country_clean" in partner_df.columns:
            s1_countries = s1_df["country_clean"].unique()
            pair_dfs = []
            for country in s1_countries:
                sub_s1 = s1_df[s1_df["country_clean"] == country]
                sub_partner = partner_df[partner_df["country_clean"] == country]
                if len(sub_partner) == 0:
                    # Fallback to entire partner pool if country not in partner
                    sub_partner = partner_df
                c_df = self._generate_candidates_for_pool(sub_s1, sub_partner)
                pair_dfs.append(c_df)
            if pair_dfs:
                return pd.concat(pair_dfs, ignore_index=True)
            return pd.DataFrame()
        else:
            return self._generate_candidates_for_pool(s1_df, partner_df)

    def _generate_candidates_for_pool(
        self,
        s1_df: pd.DataFrame,
        partner_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Runs retrievers over single partition pool."""
        all_raw_pairs = []

        partner_ids = partner_df["entity_id"].tolist()
        s1_ids = s1_df["entity_id"].tolist()

        # 1. Name Core Char TF-IDF
        r_name = TfidfRetriever("name_tfidf", analyzer="char_wb", ngram_range=(3, 5))
        r_name.fit_index(partner_df["name_core"].tolist(), partner_ids)
        pairs_name = r_name.retrieve(s1_df["name_core"].tolist(), s1_ids, top_k=self.top_k_name)
        all_raw_pairs.extend(pairs_name)

        # 2. Name + Address Char TF-IDF
        p_name_addr = (partner_df["name_clean"] + " " + partner_df["address_clean"]).tolist()
        s1_name_addr = (s1_df["name_clean"] + " " + s1_df["address_clean"]).tolist()
        r_comb = TfidfRetriever("name_addr_tfidf", analyzer="char_wb", ngram_range=(3, 5))
        r_comb.fit_index(p_name_addr, partner_ids)
        pairs_comb = r_comb.retrieve(s1_name_addr, s1_ids, top_k=self.top_k_name_addr)
        all_raw_pairs.extend(pairs_comb)

        # 3. Address Word TF-IDF
        r_addr = TfidfRetriever("addr_tfidf", analyzer="word", ngram_range=(1, 2))
        r_addr.fit_index(partner_df["address_clean"].tolist(), partner_ids)
        pairs_addr = r_addr.retrieve(s1_df["address_clean"].tolist(), s1_ids, top_k=self.top_k_addr)
        all_raw_pairs.extend(pairs_addr)

        # 4. Postal Code Exact Block
        r_postal = KeyRetriever("postal_block")
        r_postal.fit_index(partner_df["postal_code"].tolist(), partner_ids)
        pairs_postal = r_postal.retrieve(s1_df["postal_code"].tolist(), s1_ids)
        all_raw_pairs.extend(pairs_postal)

        # 5. Acronym Key Block
        r_acronym = KeyRetriever("acronym_block")
        r_acronym.fit_index(partner_df["name_acronym"].tolist(), partner_ids)
        pairs_acronym = r_acronym.retrieve(s1_df["name_acronym"].tolist(), s1_ids)
        all_raw_pairs.extend(pairs_acronym)

        # Aggregate pairs
        cand_dict: Dict[Tuple[str, str], Dict[str, float]] = {}
        for s1_id, p_id, score, ret_name in all_raw_pairs:
            pair_key = (s1_id, p_id)
            if pair_key not in cand_dict:
                cand_dict[pair_key] = {
                    "score_name_tfidf": 0.0,
                    "score_name_addr_tfidf": 0.0,
                    "score_addr_tfidf": 0.0,
                    "score_postal_block": 0.0,
                    "score_acronym_block": 0.0,
                    "max_retriever_score": 0.0
                }
            score_col = f"score_{ret_name}"
            cand_dict[pair_key][score_col] = max(cand_dict[pair_key].get(score_col, 0.0), score)
            cand_dict[pair_key]["max_retriever_score"] = max(cand_dict[pair_key]["max_retriever_score"], score)

        rows = []
        for (s1_id, p_id), scores in cand_dict.items():
            row = {"source1_entity_id": s1_id, "partner_entity_id": p_id}
            row.update(scores)
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=[
                "source1_entity_id", "partner_entity_id",
                "score_name_tfidf", "score_name_addr_tfidf", "score_addr_tfidf",
                "score_postal_block", "score_acronym_block", "max_retriever_score"
            ])

        df = pd.DataFrame(rows)

        # Cap candidates per S1 entity by max_retriever_score
        df = df.sort_values(by=["source1_entity_id", "max_retriever_score"], ascending=[True, False])
        df = df.groupby("source1_entity_id").head(self.max_candidates_per_s1).reset_index(drop=True)
        return df

    def to_candidate_dict(self, candidate_df: pd.DataFrame, all_s1_ids: List[str]) -> Dict[str, Set[str]]:
        """Converts candidate DataFrame to dict mapping source1_entity_id -> set of candidate IDs."""
        result = {s1: set() for s1 in all_s1_ids}
        if len(candidate_df) > 0:
            grouped = candidate_df.groupby("source1_entity_id")["partner_entity_id"].apply(set)
            for s1_id, partners in grouped.items():
                result[s1_id] = partners
        return result
