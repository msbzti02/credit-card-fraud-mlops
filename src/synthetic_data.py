import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_config


def generate_synthetic_samples(
    df_class: pd.DataFrame,
    n_samples: int,
    noise_scale: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:

    rng = np.random.RandomState(random_state)

    features = df_class.select_dtypes(include=[np.number]).columns.tolist()
    if "Class" in features:
        features.remove("Class")

    real_values = df_class[features].values

    idx = rng.choice(len(real_values), size=n_samples, replace=True)
    synthetic = real_values[idx].copy()

    for col_idx in range(synthetic.shape[1]):
        col_std = np.std(real_values[:, col_idx])
        noise = rng.normal(0, noise_scale * col_std, size=n_samples)
        synthetic[:, col_idx] += noise

    return pd.DataFrame(synthetic, columns=features)


def generate_augmented_dataset(
    df: pd.DataFrame,
    target_total: int = 1_000_000,
    fraud_ratio: float = 0.002,
    random_state: int = 42,
) -> pd.DataFrame:

    real_legit = df[df["Class"] == 0]
    real_fraud = df[df["Class"] == 1]

    n_real = len(df)
    n_synthetic = target_total - n_real

    if n_synthetic <= 0:
        print("[INFO] Dataset already meets target size. No augmentation needed.")
        return df

    target_fraud_total = int(target_total * fraud_ratio)
    target_legit_total = target_total - target_fraud_total

    n_synth_fraud = max(0, target_fraud_total - len(real_fraud))
    n_synth_legit = max(0, target_legit_total - len(real_legit))

    print(
        f"[INFO] Original dataset: {n_real:,} rows "
        f"({len(real_fraud)} fraud, {len(real_legit):,} legit)"
    )
    print(
        f"[INFO] Target: {target_total:,} total rows "
        f"(fraud_ratio={fraud_ratio:.2%})"
    )
    print(
        f"[INFO] Generating {n_synth_legit:,} synthetic legit + "
        f"{n_synth_fraud:,} synthetic fraud ..."
    )

    synth_legit = generate_synthetic_samples(
        real_legit,
        n_synth_legit,
        noise_scale=0.03,
        random_state=random_state,
    )
    synth_legit["Class"] = 0
    synth_legit["is_synthetic"] = 1

    synth_fraud = generate_synthetic_samples(
        real_fraud,
        n_synth_fraud,
        noise_scale=0.08,
        random_state=random_state + 1,
    )
    synth_fraud["Class"] = 1
    synth_fraud["is_synthetic"] = 1

    df_marked = df.copy()
    df_marked["is_synthetic"] = 0

    augmented = pd.concat(
        [df_marked, synth_legit, synth_fraud],
        ignore_index=True,
    )
    augmented = augmented.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    print(
        f"[INFO] Augmented dataset: {len(augmented):,} rows "
        f"({augmented['Class'].sum():,} fraud, "
        f"{(augmented['Class'] == 0).sum():,} legit)"
    )
    print(
        f"[INFO] Synthetic rows: {augmented['is_synthetic'].sum():,} "
        f"({augmented['is_synthetic'].mean():.1%})"
    )

    return augmented


def main():

    config = load_config()
    raw_path = config["data"]["raw_path"]

    print("=" * 60)
    print("  SYNTHETIC DATA GENERATION")
    print("=" * 60)

    print(f"\n[INFO] Loading original data from {raw_path} ...")
    df = pd.read_csv(raw_path)

    augmented = generate_augmented_dataset(
        df,
        target_total=1_000_000,
        fraud_ratio=0.002,
        random_state=config["data"]["random_state"],
    )

    out_dir = os.path.dirname(raw_path)
    aug_path = os.path.join(out_dir, "creditcard_augmented.csv")
    augmented.to_csv(aug_path, index=False)
    file_size_mb = os.path.getsize(aug_path) / (1024 * 1024)
    print(f"\n[INFO] Saved augmented dataset to {aug_path}")
    print(f"[INFO] File size: {file_size_mb:.1f} MB")

    print(f"\n{'='*60}")
    print(f"  AUGMENTATION COMPLETE")
    print(f"  Original:  {len(df):>10,} rows")
    print(f"  Augmented: {len(augmented):>10,} rows")
    print(f"  Growth:    {len(augmented)/len(df):.1f}x")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
