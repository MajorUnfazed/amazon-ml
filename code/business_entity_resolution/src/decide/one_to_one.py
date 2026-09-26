"""Global assignment consistency layer for LinkSure.
Enforces that each S2/S3 record is claimed by at most one S1 reference entity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple

def resolve_one_to_one(
    scored_pairs_df: pd.DataFrame,
    prob_col: str = "p_cal",
    margin: float = 0.0
) -> pd.DataFrame:
    """
    If multiple S1 entities retrieve the same partner entity,
    assigns the partner entity exclusively to the S1 entity with the highest probability.
    Drops the partner from all competing S1 entities.
    """
    if len(scored_pairs_df) == 0:
        return scored_pairs_df

    # Sort descending by probability
    sorted_df = scored_pairs_df.sort_values(by=prob_col, ascending=False).copy()

    # Find the maximum probability for each partner (already sorted descending)
    best_claims = sorted_df.drop_duplicates(subset=["partner_entity_id"], keep="first")
    return best_claims.reset_index(drop=True)
