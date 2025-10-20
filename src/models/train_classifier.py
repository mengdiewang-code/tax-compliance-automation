
import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from ..preprocessing.cleaning import clean_df
from ..preprocessing.features import make_features

def main():
    df = pd.read_csv('data/labeled_training_data.csv')
    df = clean_df(df)
    df, feat_cols = make_features(df)
    X = df[feat_cols].values
    y = df['non_compliant'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42)
    clf.fit(X_train, y_train)
    os.makedirs('results/models', exist_ok=True)
    joblib.dump({"model": clf, "features": feat_cols}, 'results/models/xgb_classifier.joblib')
    report = classification_report(y_test, clf.predict(X_test), output_dict=True)
    pd.DataFrame(report).to_csv('results/classifier_report.csv')
    print('Saved results/models/xgb_classifier.joblib')

if __name__ == '__main__':
    main()
