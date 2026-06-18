import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    v_cols = [c for c in df.columns if c.startswith("V")]
    n_new = 0

    if "Amount" in df.columns:

        df["amount_log"] = np.log1p(df["Amount"])
        n_new += 1

        amt_mean = df["Amount"].mean()
        amt_std = df["Amount"].std()
        if amt_std > 0:
            df["amount_zscore"] = (df["Amount"] - amt_mean) / amt_std
        else:
            df["amount_zscore"] = 0.0
        n_new += 1

        threshold = df["Amount"].quantile(0.95)
        df["is_high_amount"] = (df["Amount"] > threshold).astype(int)
        n_new += 1

        df["amount_bin"] = pd.qcut(df["Amount"], q=4, labels=False, duplicates="drop")
        n_new += 1

    if "Time" in df.columns:

        df["time_hours"] = df["Time"] / 3600.0
        n_new += 1

        df["hour_of_day"] = (df["Time"] % 86400) / 3600.0
        n_new += 1

        hour = df["hour_of_day"]
        df["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)
        n_new += 1

        df["day_segment"] = pd.cut(
            df["hour_of_day"],
            bins=[0, 6, 12, 18, 24],
            labels=[3, 0, 1, 2],
            include_lowest=True,
        ).astype(int)
        n_new += 1

    if len(v_cols) > 0:
        v_data = df[v_cols]

        df["v_mean"] = v_data.mean(axis=1)
        df["v_std"] = v_data.std(axis=1).fillna(0)
        df["v_min"] = v_data.min(axis=1)
        df["v_max"] = v_data.max(axis=1)
        df["v_range"] = df["v_max"] - df["v_min"]
        df["v_skew"] = v_data.skew(axis=1).fillna(0)
        df["v_kurtosis"] = v_data.kurtosis(axis=1).fillna(0)
        n_new += 7

        df["v_extreme_count"] = (v_data.abs() > 2).sum(axis=1)
        n_new += 1

    interaction_pairs = [
        ("V14", "V12"),
        ("V17", "V14"),
        ("V10", "V12"),
        ("V4", "V11"),
        ("V3", "V7"),
    ]
    for f1, f2 in interaction_pairs:
        if f1 in df.columns and f2 in df.columns:
            df[f"interact_{f1}_{f2}"] = df[f1] * df[f2]
            n_new += 1

    poly_features = ["V4", "V14", "V12", "V17", "V10"]
    for feat in poly_features:
        if feat in df.columns:
            df[f"{feat}_squared"] = df[feat] ** 2
            n_new += 1

    abs_features = ["V1", "V2", "V3", "V4", "V14"]
    for feat in abs_features:
        if feat in df.columns:
            df[f"{feat}_abs"] = df[feat].abs()
            n_new += 1

    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"[INFO] Filling {nan_count} NaN values with 0")
        df = df.fillna(0)

    print(
        f"[INFO] Feature engineering: added {n_new} new features "
        f"({len(df.columns)} total columns)"
    )

    return df


def main():

    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_preprocessing import load_config, load_data

    config = load_config()
    df = load_data(config)

    print(f"\nBefore: {df.shape[1]} columns")
    df_enhanced = engineer_features(df)
    print(f"After:  {df_enhanced.shape[1]} columns")

    original_cols = set(["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)])
    new_cols = [c for c in df_enhanced.columns if c not in original_cols]
    print(f"\nNew features ({len(new_cols)}):")
    for col in new_cols:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
