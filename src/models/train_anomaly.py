
import os, joblib, pandas as pd
from sklearn.ensemble import IsolationForest
from ..preprocessing.cleaning import clean_df
from ..preprocessing.features import make_features

def main():
    df = pd.read_csv('data/sample_transactions.csv')
    df = clean_df(df)
    df, feat_cols = make_features(df)
    X = df[feat_cols].values
    iforest = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iforest.fit(X)
    os.makedirs('results/models', exist_ok=True)
    joblib.dump({"model": iforest, "features": feat_cols}, 'results/models/iforest.joblib')
    print('Saved results/models/iforest.joblib')

if __name__ == '__main__':
    main()
