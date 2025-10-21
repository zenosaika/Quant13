from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["returns"] = enriched["close"].pct_change()
    enriched["log_returns"] = (enriched["close"] / enriched["close"].shift(1)).apply(
        lambda x: 0.0 if pd.isna(x) or x <= 0 else float(np.log(x))
    )
    return enriched
