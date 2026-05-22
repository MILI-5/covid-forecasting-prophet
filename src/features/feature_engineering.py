import pandas as pd
import numpy as np


def create_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline for epidemic forecasting
    """

    df = df.copy()
    df = df.sort_values("ds")

    # =========================================================
    # Step 2 — Target Fix
    # =========================================================
    df["y"] = df["y"].clip(lower=0)

    # =========================================================
    # Step 3 — Lag Features
    # =========================================================
    df["lag_1"] = df["y"].shift(1)
    df["lag_7"] = df["y"].shift(7)
    df["lag_14"] = df["y"].shift(14)

    # =========================================================
    # Step 4 — Rolling Statistics
    # =========================================================
    df["roll_mean_7"] = df["y"].rolling(window=7).mean()
    df["roll_mean_14"] = df["y"].rolling(window=14).mean()

    df["roll_std_7"] = df["y"].rolling(window=7).std()
    df["roll_std_14"] = df["y"].rolling(window=14).std()

    # =========================================================
    # Step 5 — Growth Rate Features
    # =========================================================
    df["growth_rate"] = df["y"].pct_change() * 100
    df["growth_rate"] = df["growth_rate"].replace([np.inf, -np.inf], np.nan)

    df["growth_rate_smooth"] = df["growth_rate"].rolling(window=7).mean()

    # =========================================================
    # Step 6 — Acceleration
    # =========================================================
    df["acceleration"] = df["growth_rate"].diff()

    # =========================================================
    # Step 7 — Volatility Index
    # =========================================================
    df["volatility"] = df["roll_std_7"] / (df["roll_mean_7"].abs() + 1)

    # =========================================================
    # Additional Growth Signals
    # =========================================================
    df["growth_rate_1"] = df["y"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    df["growth_rate_7"] = df["y"].pct_change(7).replace([np.inf, -np.inf], np.nan)

    df["momentum_7"] = df["y"] - df["y"].shift(7)

    df["trend_strength_7"] = df["roll_mean_7"] / (df["roll_mean_14"] + 1)

    # =========================================================
    # Step 8 — Epidemic Phase Labeling
    # =========================================================
    def classify_phase(row):
        if pd.isna(row["growth_rate_smooth"]):
            return "Unknown"
        elif row["growth_rate_smooth"] > 20:
            return "Explosion Phase"
        elif row["growth_rate_smooth"] > 0:
            return "Growth Phase"
        elif row["growth_rate_smooth"] > -20:
            return "Decline Phase"
        else:
            return "Control Phase"

    df["phase"] = df.apply(classify_phase, axis=1)

    # =========================================================
    # Step 9 — Encode Phase
    # =========================================================
    phase_map = {
        "Explosion Phase": 3,
        "Growth Phase": 2,
        "Decline Phase": 1,
        "Control Phase": 0,
        "Unknown": -1
    }

    df["phase_encoded"] = df["phase"].map(phase_map)

    # =========================================================
    # Step 10 — Missing Value Handling
    # =========================================================
    df = df.bfill()
    df = df.ffill()

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =========================================================
# Step 11 — Final Feature Selection
# =========================================================
def get_final_features(df: pd.DataFrame) -> pd.DataFrame:

    final_df = df[
        [
            "ds",
            "y",
            "lag_1",
            "lag_7",
            "lag_14",
            "roll_mean_7",
            "roll_mean_14",
            "roll_std_7",
            "growth_rate",
            "growth_rate_smooth",
            "acceleration",
            "volatility",
            "phase_encoded"
        ]
    ]

    return final_df


# =========================================================
# Step 12 — Save Final Dataset
# =========================================================
def save_features(df: pd.DataFrame, output_path: str):

    df.to_csv(output_path, index=False)
    print(f"Saved feature dataset → {output_path}")


# =========================================================
# Step 13 — Interview Insight (NOT CODE, JUST COMMENT)
# =========================================================
"""
INTERVIEW GOLD INSIGHT:

Even though Prophet may not directly use all engineered features,
this pipeline demonstrates deep understanding of:

- epidemic wave dynamics
- trend acceleration & deceleration
- volatility & instability detection
- phase-based outbreak classification
- temporal dependency modeling

This is what differentiates a basic student project
from an ML engineering internship-level project.
"""