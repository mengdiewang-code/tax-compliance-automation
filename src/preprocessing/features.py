
import pandas as pd
import numpy as np

CATS = ["category","country","method","vendor"]

def encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATS:
        if c in df.columns:
            df[c+"_idx"] = df[c].astype("category").cat.codes
    return df

def make_features(df: pd.DataFrame):
    df = encode_cats(df)
    df["vat_ratio"] = np.where(df["amount"]>0, df["vat_claimed"]/df["amount"], 0.0)
    df["cat_amount_mean"] = df.groupby("category")["amount"].transform("mean")
    df["cat_amount_std"] = df.groupby("category")["amount"].transform("std").replace(0,1)
    df["amount_z"] = (df["amount"] - df["cat_amount_mean"]) / df["cat_amount_std"]
    df["is_round_100"] = ((df["amount"]%100)==0).astype(int)
    df["is_round_50"] = ((df["amount"]%50)==0).astype(int)
    feat_cols = ["amount","vat_claimed","vat_ratio","hour","dayofweek","amount_z","is_round_100","is_round_50",
                 "category_idx","country_idx","method_idx","vendor_idx"]
    return df, feat_cols
