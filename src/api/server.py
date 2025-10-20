
from flask import Flask, request, jsonify
import joblib, yaml, pandas as pd
from datetime import datetime
from ..preprocessing.cleaning import clean_df
from ..preprocessing.features import make_features
from ..models.risk_scorer import apply_rules, load_watchlist

app = Flask(__name__)

xgb = None
iforest = None
cfg = yaml.safe_load(open('data/rules.yml'))
watchlist = load_watchlist()

def lazy_models():
    global xgb, iforest
    if xgb is None:
        xgb = joblib.load('results/models/xgb_classifier.joblib')
    if iforest is None:
        iforest = joblib.load('results/models/iforest.joblib')
    return xgb, iforest

@app.get("/health")
def health():
    return {"status":"ok","time": datetime.utcnow().isoformat()+"Z"}

def score_one(d):
    xgb, iforest = lazy_models()
    df = pd.DataFrame([d])
    df = clean_df(df)
    df, feat_cols = make_features(df)
    row = df.iloc[0]
    clf, fcols = xgb['model'], xgb['features']
    ifm, fcols_if = iforest['model'], iforest['features']

    X = row[fcols].values.reshape(1,-1)
    p_non = float(clf.predict_proba(X)[0,1])
    X2 = row[fcols_if].values.reshape(1,-1)
    ano = float(-ifm.decision_function(X2)[0])

    rscore, reasons = apply_rules(row, cfg['rules'], watchlist)

    risk = cfg['weights']['rules']*rscore + cfg['weights']['anomaly']*ano + cfg['weights']['classifier']*p_non
    level = "ok"
    if risk >= cfg['thresholds']['alert']:
        level = "alert"
    elif risk >= cfg['thresholds']['review']:
        level = "review"
    return {"risk_score": float(risk), "level": level, "reasons": reasons, "components": {"rules": rscore, "anomaly": ano, "classifier_p": p_non}}

@app.post("/score")
def score():
    d = request.json or {}
    return jsonify(score_one(d))

@app.post("/batch_score")
def batch_score():
    items = request.json or []
    out = [score_one(d) for d in items]
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
