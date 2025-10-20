
import argparse, json, yaml, joblib, pandas as pd
from datetime import datetime
from ..preprocessing.cleaning import clean_df
from ..preprocessing.features import make_features

def load_watchlist(path='data/watchlist.csv'):
    try:
        wl = pd.read_csv(path)
        return set(wl['entity'].astype(str).str.lower().tolist())
    except Exception:
        return set()

def apply_rules(row, rules, watchlist):
    score = 0.0
    reasons = []
    env = row.to_dict()
    env['watchlist'] = watchlist
    for r in rules:
        cond = r.get('condition','')
        try:
            if eval(cond, {}, env):
                score += float(r.get('score',0))
                reasons.append(r['id'])
        except Exception:
            continue
    return score, reasons

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--rules', default='data/rules.yml')
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.rules))
    w_rules = float(cfg['weights']['rules'])
    w_ano = float(cfg['weights']['anomaly'])
    w_cls = float(cfg['weights']['classifier'])
    thr_alert = float(cfg['thresholds']['alert'])
    thr_review = float(cfg['thresholds']['review'])
    rules = cfg['rules']

    df = pd.read_csv(args.input)
    df = clean_df(df)
    df, feat_cols = make_features(df)

    xgb = joblib.load('results/models/xgb_classifier.joblib')
    clf, fcols_clf = xgb['model'], xgb['features']
    iforest = joblib.load('results/models/iforest.joblib')
    ifm, fcols_if = iforest['model'], iforest['features']

    watchlist = load_watchlist()

    preds = []
    for _, row in df.iterrows():
        X = row[fcols_clf].values.reshape(1,-1)
        p_non = float(clf.predict_proba(X)[0,1])

        X2 = row[fcols_if].values.reshape(1,-1)
        ano = float(-ifm.decision_function(X2)[0])

        rscore, reasons = apply_rules(row, rules, watchlist)

        risk = w_rules*rscore + w_ano*ano + w_cls*p_non
        level = "ok"
        if risk >= thr_alert:
            level = "alert"
        elif risk >= thr_review:
            level = "review"

        preds.append({
            "transaction_id": int(row.get("transaction_id",-1)),
            "timestamp": str(row.get("timestamp")),
            "vendor": row.get("vendor"),
            "amount": float(row.get("amount",0)),
            "risk_score": round(risk,4),
            "components": {"rules": rscore, "anomaly": ano, "classifier_p": p_non},
            "level": level,
            "reasons": reasons
        })

    out = {"generated_at": datetime.utcnow().isoformat()+"Z", "count": len(preds), "alerts": preds}
    with open(args.output,"w") as f: json.dump(out, f, indent=2)
    print(f"Saved alerts to {args.output}")

if __name__ == '__main__':
    main()
