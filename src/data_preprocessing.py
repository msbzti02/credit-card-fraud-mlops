import os

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str = "configs/config.yaml") -> dict:

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(config: dict, use_augmented: bool = False) -> pd.DataFrame:

    raw_path = config["data"]["raw_path"]

    if use_augmented:
        aug_path = raw_path.replace("creditcard.csv", "creditcard_augmented.csv")
        if os.path.exists(aug_path):
            print(f"[INFO] Loading AUGMENTED data from {aug_path} ...")
            df = pd.read_csv(aug_path)

            if "is_synthetic" in df.columns:
                df = df.drop(columns=["is_synthetic"])
            print(f"[INFO] Dataset shape: {df.shape}")
            print(f"[INFO] Fraud ratio: {df['Class'].mean():.4%}")
            return df
        else:
            print("[WARN] Augmented dataset not found, using original.")

    print(f"[INFO] Loading data from {raw_path} ...")
    df = pd.read_csv(raw_path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Fraud ratio: {df['Class'].mean():.4%}")
    return df


def preprocess_data(
    df: pd.DataFrame,
    config: dict,
    apply_smote: bool = True,
    save: bool = True,
    apply_feature_engineering: bool = True,
) -> tuple:

    random_state = config["data"]["random_state"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]

    if apply_feature_engineering and "Amount" in df.columns:
        try:
            from src.feature_engineering import engineer_features

            df = engineer_features(df)
        except ImportError:
            print("[WARN] Feature engineering module not found, skipping.")

    scaler = StandardScaler()
    if "Amount" in df.columns:
        df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
        df = df.drop(columns=["Amount"])
    if "Time" in df.columns:
        df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
        df = df.drop(columns=["Time"])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=y,
    )

    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - relative_val,
        random_state=random_state,
        stratify=y_temp,
    )

    print(f"[INFO] Train shape : {X_train.shape}  (fraud={y_train.sum()})")
    print(f"[INFO] Val shape   : {X_val.shape}  (fraud={y_val.sum()})")
    print(f"[INFO] Test shape  : {X_test.shape}  (fraud={y_test.sum()})")

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[INFO] After SMOTE : {X_train.shape}  (fraud={y_train.sum()})")

    if save:
        out_dir = config["data"]["processed_dir"]
        os.makedirs(out_dir, exist_ok=True)
        X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
        X_val.to_csv(f"{out_dir}/X_val.csv", index=False)
        X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
        y_val.to_csv(f"{out_dir}/y_val.csv", index=False)
        y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
        print(f"[INFO] Processed data saved to {out_dir}/")

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_processed_data(config: dict) -> tuple:

    d = config["data"]["processed_dir"]
    X_train = pd.read_csv(f"{d}/X_train.csv")
    X_val = pd.read_csv(f"{d}/X_val.csv")
    X_test = pd.read_csv(f"{d}/X_test.csv")
    y_train = pd.read_csv(f"{d}/y_train.csv").squeeze()
    y_val = pd.read_csv(f"{d}/y_val.csv").squeeze()
    y_test = pd.read_csv(f"{d}/y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    cfg = load_config()
    df = load_data(cfg)
    preprocess_data(df, cfg, apply_smote=True, save=True)
    print("[INFO] Data preprocessing complete.")
