
import argparse, time, json, yaml, joblib, pandas as pd
from datetime import datetime
from ..preprocessing.cleaning import clean_df
from ..preprocessing.features import make_features
from ..models.risk_scorer import apply_rules, load_watchlist

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rate', type=int, default=5, help='events per minute')
    p.add_argument('--source', default='data/sample_transactions.csv')
    p.add_argument('--outdir', default='results/alerts')
    args = p.parse_args()

    cfg = yaml.safe_load(open('data/rules.yml'))
    xgb = joblib.load('results/models/xgb_classifier.joblib')
    iforest = joblib.load('results/models/iforest.joblib')
    watchlist = load_watchlist()

    df = pd.read_csv(args.source)
    df = clean_df(df)
    df, feat_cols = make_features(df)

    per_event_sleep = max(0.1, 60.0/args.rate)

    for i, row in df.iterrows():
        X = row[xgb['features']].values.reshape(1,-1)
        p_non = float(xgb['model'].predict_proba(X)[0,1])
        X2 = row[iforest['features']].values.reshape(1,-1)
        ano = float(-iforest['model'].decision_function(X2)[0])
        rscore, reasons = apply_rules(row, cfg['rules'], watchlist)
        risk = cfg['weights']['rules']*rscore + cfg['weights']['anomaly']*ano + cfg['weights']['classifier']*p_non

        level = "ok"
        if risk >= cfg['thresholds']['alert']:
            level = "alert"
        elif risk >= cfg['thresholds']['review']:
            level = "review"

        alert = {
            "ts": datetime.utcnow().isoformat()+"Z",
            "transaction_id": int(row.get("transaction_id",-1)),
            "vendor": row.get("vendor"),
            "amount": float(row.get("amount",0)),
            "level": level,
            "risk_score": round(float(risk),4),
            "reasons": reasons
        }
        outpath = f"{args.outdir}/stream_alert_{i:05d}.json"
        with open(outpath, "w") as f:
            json.dump(alert, f)
        print(f"[{level.upper()}] -> {outpath}")
        time.sleep(per_event_sleep)

if __name__ == '__main__':
    main()
