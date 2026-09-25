"""Feature Engineering Engine for LinkSure Business Entity Resolution."""

import math
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

class FeatureBuilder:
    """
    Constructs rich, country-agnostic pair features for (S1, Candidate) pairs.
    """
    def __init__(self):
        self.idf_dict: Dict[str, float] = {}

    def fit_idf(self, corpus_texts: List[str]):
        """Fits inverse document frequency dictionary on name and address words."""
        doc_count = len(corpus_texts)
        if doc_count == 0:
            return

        df_counter: Counter = Counter()
        for text in corpus_texts:
            tokens = set(str(text).lower().split())
            df_counter.update(tokens)

        # Standard smoothed IDF: log((N + 1) / (df + 1)) + 1
        self.idf_dict = {
            w: math.log((doc_count + 1.0) / (df + 1.0)) + 1.0
            for w, df in df_counter.items()
        }

    def get_token_idf(self, token: str) -> float:
        """Returns token IDF or default for unknown words."""
        return self.idf_dict.get(token.lower(), 1.0)

    def build_features(
        self,
        candidate_df: pd.DataFrame,
        s1_norm_df: pd.DataFrame,
        partner_norm_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Takes candidate pairs DataFrame and normalized records DataFrames.
        Computes ~45 relational features per pair.
        """
        if len(candidate_df) == 0:
            return pd.DataFrame()

        # Build fast lookup dictionaries
        s1_lookup = s1_norm_df.set_index("entity_id").to_dict(orient="index")
        partner_lookup = partner_norm_df.set_index("entity_id").to_dict(orient="index")

        feature_rows = []

        for _, row in candidate_df.iterrows():
            s1_id = row["source1_entity_id"]
            p_id = row["partner_entity_id"]

            s1 = s1_lookup.get(s1_id, {})
            p = partner_lookup.get(p_id, {})

            feat = {
                "source1_entity_id": s1_id,
                "partner_entity_id": p_id
            }

            # 1. Retriever scores
            feat["score_name_tfidf"] = float(row.get("score_name_tfidf", 0.0))
            feat["score_name_addr_tfidf"] = float(row.get("score_name_addr_tfidf", 0.0))
            feat["score_addr_tfidf"] = float(row.get("score_addr_tfidf", 0.0))
            feat["score_postal_block"] = float(row.get("score_postal_block", 0.0))
            feat["score_acronym_block"] = float(row.get("score_acronym_block", 0.0))
            feat["max_retriever_score"] = float(row.get("max_retriever_score", 0.0))

            # S1 attributes
            s1_clean = s1.get("name_clean", "")
            s1_core = s1.get("name_core", "")
            s1_suffix = s1.get("legal_suffix", "")
            s1_addr = s1.get("address_clean", "")
            s1_postal = s1.get("postal_code", "")
            s1_nums = set(s1.get("building_numbers", []))
            s1_lms = set(s1.get("landmarks", []))
            s1_acr = s1.get("name_acronym", "")
            s1_vars = s1.get("name_variants", [s1_clean])

            # Partner attributes
            p_clean = p.get("name_clean", "")
            p_core = p.get("name_core", "")
            p_suffix = p.get("legal_suffix", "")
            p_addr = p.get("address_clean", "")
            p_postal = p.get("postal_code", "")
            p_nums = set(p.get("building_numbers", []))
            p_lms = set(p.get("landmarks", []))
            p_acr = p.get("name_acronym", "")
            p_vars = p.get("name_variants", [p_clean])

            # Source origin
            feat["is_source2"] = 1.0 if str(p_id).startswith("S2-") else 0.0

            # 2. Name string similarities (Clean name)
            feat["name_clean_jaro_winkler"] = float(distance.JaroWinkler.similarity(s1_clean, p_clean))
            feat["name_clean_levenshtein"] = float(fuzz.ratio(s1_clean, p_clean) / 100.0)
            feat["name_clean_token_sort"] = float(fuzz.token_sort_ratio(s1_clean, p_clean) / 100.0)
            feat["name_clean_token_set"] = float(fuzz.token_set_ratio(s1_clean, p_clean) / 100.0)
            feat["name_clean_partial"] = float(fuzz.partial_ratio(s1_clean, p_clean) / 100.0)

            # 3. Name string similarities (Core name)
            feat["name_core_jaro_winkler"] = float(distance.JaroWinkler.similarity(s1_core, p_core))
            feat["name_core_levenshtein"] = float(fuzz.ratio(s1_core, p_core) / 100.0)
            feat["name_core_token_sort"] = float(fuzz.token_sort_ratio(s1_core, p_core) / 100.0)
            feat["name_core_token_set"] = float(fuzz.token_set_ratio(s1_core, p_core) / 100.0)

            # DBA best variant match
            best_dba = max([fuzz.ratio(v1, v2) / 100.0 for v1 in s1_vars for v2 in p_vars], default=0.0)
            feat["name_dba_best_ratio"] = float(best_dba)

            # Name length ratio
            len_s1, len_p = len(s1_core), len(p_core)
            max_len = max(len_s1, len_p)
            min_len = min(len_s1, len_p)
            feat["name_length_ratio"] = float(min_len / max_len) if max_len > 0 else 1.0
            feat["name_length_diff"] = float(abs(len_s1 - len_p))

            # Acronym match
            feat["acronym_exact_match"] = 1.0 if (s1_acr and p_acr and s1_acr == p_acr) else 0.0

            # Legal suffix match/conflict
            if s1_suffix and p_suffix:
                feat["suffix_status"] = 1.0 if (s1_suffix == p_suffix) else -1.0
            else:
                feat["suffix_status"] = 0.0

            # 4. Token rarity features (Core name)
            s1_tokens = set(s1_core.split())
            p_tokens = set(p_core.split())
            shared_name_tokens = s1_tokens & p_tokens
            unshared_s1_tokens = s1_tokens - p_tokens
            unshared_p_tokens = p_tokens - s1_tokens
            unshared_tokens = unshared_s1_tokens | unshared_p_tokens

            shared_idfs = [self.get_token_idf(t) for t in shared_name_tokens]
            unshared_idfs = [self.get_token_idf(t) for t in unshared_tokens]

            feat["token_shared_count"] = float(len(shared_name_tokens))
            feat["token_shared_idf_sum"] = float(sum(shared_idfs))
            feat["token_shared_max_idf"] = float(max(shared_idfs)) if shared_idfs else 0.0
            feat["token_unshared_max_idf"] = float(max(unshared_idfs)) if unshared_idfs else 0.0
            feat["token_jaccard"] = float(len(shared_name_tokens) / len(s1_tokens | p_tokens)) if (s1_tokens | p_tokens) else 0.0

            # 5. Address similarities
            feat["addr_token_set"] = float(fuzz.token_set_ratio(s1_addr, p_addr) / 100.0)
            feat["addr_token_sort"] = float(fuzz.token_sort_ratio(s1_addr, p_addr) / 100.0)
            feat["addr_levenshtein"] = float(fuzz.ratio(s1_addr, p_addr) / 100.0)
            feat["addr_partial"] = float(fuzz.partial_ratio(s1_addr, p_addr) / 100.0)

            s1_addr_toks = set(s1_addr.split())
            p_addr_toks = set(p_addr.split())
            shared_addr = s1_addr_toks & p_addr_toks
            feat["addr_token_jaccard"] = float(len(shared_addr) / len(s1_addr_toks | p_addr_toks)) if (s1_addr_toks | p_addr_toks) else 0.0

            # 6. Postal code features
            if s1_postal and p_postal:
                if s1_postal == p_postal:
                    feat["postal_match"] = 1.0
                    feat["postal_prefix_match"] = 1.0
                else:
                    feat["postal_match"] = -1.0
                    feat["postal_prefix_match"] = 1.0 if s1_postal[:3] == p_postal[:3] else -1.0
            else:
                feat["postal_match"] = 0.0
                feat["postal_prefix_match"] = 0.0

            # 7. Building number match
            if s1_nums and p_nums:
                feat["building_num_match"] = 1.0 if (s1_nums & p_nums) else -1.0
            else:
                feat["building_num_match"] = 0.0

            # 8. Landmark match
            if s1_lms or p_lms:
                shared_lms = s1_lms & p_lms
                feat["landmark_shared"] = 1.0 if shared_lms else 0.0
            else:
                feat["landmark_shared"] = 0.0

            feature_rows.append(feat)

        out_df = pd.DataFrame(feature_rows)

        # 9. Context and rank features
        out_df = self._add_context_and_rank_features(out_df)
        return out_df

    def _add_context_and_rank_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds per-S1 ranking, score gaps, reverse ranks, and mutual best match indicators.
        """
        # Primary sort metric: blend of max_retriever_score and name similarity
        df["_rank_score"] = (
            df["max_retriever_score"] * 0.5 +
            df["name_core_jaro_winkler"] * 0.3 +
            df["addr_token_set"] * 0.2
        )

        # 1. Rank within S1
        df["rank_within_s1"] = df.groupby("source1_entity_id")["_rank_score"].rank(ascending=False, method="min")
        best_score_per_s1 = df.groupby("source1_entity_id")["_rank_score"].transform("max")
        df["gap_to_best_s1_score"] = best_score_per_s1 - df["_rank_score"]

        # Number of candidates for S1
        df["num_candidates_s1"] = df.groupby("source1_entity_id")["partner_entity_id"].transform("count")

        # 2. Reverse rank: rank of S1 among all entities claiming this partner
        df["reverse_rank_for_partner"] = df.groupby("partner_entity_id")["_rank_score"].rank(ascending=False, method="min")
        df["num_claims_for_partner"] = df.groupby("partner_entity_id")["source1_entity_id"].transform("count")

        # Mutual best match flag
        df["is_mutual_best_match"] = (
            (df["rank_within_s1"] == 1.0) & (df["reverse_rank_for_partner"] == 1.0)
        ).astype(float)

        df = df.drop(columns=["_rank_score"])
        return df
