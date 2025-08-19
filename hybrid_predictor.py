# hybrid_predictor.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")
STRENGTH_MODEL_PATH = os.path.join(MODEL_DIR, "models_strength.joblib")
WEIGHT_MODEL_PATH = os.path.join(MODEL_DIR, "models_weight.joblib")

models_strength = None
models_weight = None

# ----------------------
# Embedded lab data (from your PDF)
# ----------------------
def _build_df_from_pdf():
    rows = []
    # Control
    control = {
        0.00: {7:26.53, 28:33.33, 56:47.14},
        0.02: {7:31.29, 28:35.37, 56:41.54},
        0.04: {7:32.65, 28:38.09, 56:42.12},
        0.06: {7:36.05, 28:42.21, 56:52.13},
        0.08: {7:33.33, 28:36.05, 56:46.14},
        0.10: {7:31.29, 28:33.33, 56:44.13},
    }
    for go, dvals in control.items():
        for d,s in dvals.items():
            rows.append(("control", go, d, s, 0.0))

    # Sulphate
    sulphate = {
        0.00: {7:23.9, 28:30.0, 56:32.2},
        0.02: {7:28.2, 28:31.8, 56:35.0},
        0.04: {7:29.4, 28:34.3, 56:37.1},
        0.06: {7:32.4, 28:38.0, 56:40.9},
        0.08: {7:30.0, 28:32.4, 56:35.0},
        0.10: {7:28.2, 28:30.0, 56:31.9},
    }
    sulphate_wt = {
        0.00: {7:2.8, 28:10.7, 56:13.6},
        0.02: {7:2.4, 28:10.2, 56:12.9},
        0.04: {7:2.4, 28:11.2, 56:13.4},
        0.06: {7:2.5, 28:11.1, 56:13.3},
        0.08: {7:2.4, 28:10.8, 56:13.1},
        0.10: {7:2.9, 28:10.3, 56:13.2},
    }
    for go, dvals in sulphate.items():
        for d,s in dvals.items():
            rows.append(("sulphate", go, d, s, sulphate_wt[go][d]))

    # Acid
    acid = {
        0.00: {7:30.5, 28:35.8, 56:32.2},
        0.02: {7:31.2, 28:36.5, 56:35.0},
        0.04: {7:32.0, 28:37.1, 56:36.0},
        0.06: {7:32.8, 28:38.0, 56:37.5},
        0.08: {7:31.5, 28:36.0, 56:34.0},
        0.10: {7:30.0, 28:34.5, 56:32.0},
    }
    acid_wt = {
        0.00: {7:2.9, 28:8.7, 56:12.7},
        0.02: {7:2.9, 28:9.5, 56:12.6},
        0.04: {7:2.4, 28:9.6, 56:12.4},
        0.06: {7:2.1, 28:5.6, 56:10.8},
        0.08: {7:1.7, 28:8.7, 56:10.1},
        0.10: {7:0.2, 28:6.5, 56:7.3},
    }
    for go, dvals in acid.items():
        for d,s in dvals.items():
            rows.append(("acid", go, d, s, acid_wt[go][d]))

    # Chloride
    chloride = {
        0.00: {7:23.1, 28:29.75, 56:40.1},
        0.02: {7:26.85, 28:32.3, 56:44.5},
        0.04: {7:28.7, 28:35.2, 56:46.75},
        0.06: {7:30.1, 28:37.9, 56:50.5},
        0.08: {7:27.2, 28:34.8, 56:46.1},
        0.10: {7:26.1, 28:32.85, 56:43.2},
    }
    chloride_wt = {
        0.00: {7:0.52, 28:1.9, 56:2.83},
        0.02: {7:0.49, 28:1.68, 56:2.53},
        0.04: {7:0.44, 28:1.46, 56:2.22},
        0.06: {7:0.40, 28:1.37, 56:2.08},
        0.08: {7:0.46, 28:1.42, 56:2.2},
        0.10: {7:0.51, 28:1.47, 56:2.25},
    }
    for go, dvals in chloride.items():
        for d,s in dvals.items():
            rows.append(("chloride", go, d, s, chloride_wt[go][d]))

    df = pd.DataFrame(rows, columns=["attack", "go", "day", "strength", "weight_loss_pct"]) 
    df["go_pct"] = df["go"]
    df["day_norm"] = df["day"]
    return df

# Train models
def _train_and_save_models():
    df = _build_df_from_pdf()
    attacks = df["attack"].unique()
    models_s = {}
    models_w = {}
    for attack in attacks:
        dfa = df[df["attack"] == attack]
        X = dfa[["go_pct", "day_norm"]].values
        y_s = dfa["strength"].values
        y_w = dfa["weight_loss_pct"].values
        model_s = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
        model_w = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=1)
        model_s.fit(X, y_s)
        model_w.fit(X, y_w)
        models_s[attack] = model_s
        models_w[attack] = model_w
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(models_s, STRENGTH_MODEL_PATH)
    joblib.dump(models_w, WEIGHT_MODEL_PATH)
    return models_s, models_w

# Load or train
if os.path.exists(STRENGTH_MODEL_PATH) and os.path.exists(WEIGHT_MODEL_PATH):
    try:
        models_strength = joblib.load(STRENGTH_MODEL_PATH)
        models_weight = joblib.load(WEIGHT_MODEL_PATH)
    except Exception:
        models_strength, models_weight = _train_and_save_models()
else:
    models_strength, models_weight = _train_and_save_models()

# ----------------------
# Public API
# ----------------------
def predict_strength_curve(go_pct, attack, days_max=365, step=1, cap_loss=0.30):
    attack = attack.lower()
    days = np.arange(1, days_max+1, step)
    model = models_strength.get(attack)
    Xpred = np.column_stack([np.full_like(days, go_pct, dtype=float), days])
    preds = model.predict(Xpred)
    preds = np.clip(preds, 0.0, 200.0)
    return days, preds

def predict_weight_curve(go_pct, attack, days_max=365, step=1):
    attack = attack.lower()
    days = np.arange(1, days_max+1, step)
    model = models_weight.get(attack)
    Xpred = np.column_stack([np.full_like(days, go_pct, dtype=float), days])
    preds = model.predict(Xpred)
    preds = np.clip(preds, 0.0, 100.0)
    return days, preds

def predict_single(go_pct, days, attack, return_weight=False):
    _, s = predict_strength_curve(go_pct, attack, days_max=days, step=days)
    if return_weight:
        _, w = predict_weight_curve(go_pct, attack, days_max=days, step=days)
        return float(np.round(s[-1], 2)), float(np.round(w[-1], 2))
    return float(np.round(s[-1], 2))

if __name__ == '__main__':
    days, strengths = predict_strength_curve(0.06, 'sulphate', days_max=365)
    _, weights = predict_weight_curve(0.06, 'sulphate', days_max=365)
    out = pd.DataFrame({'day': days, 'strength_MPa': strengths, 'weight_loss_pct': weights})
    out.to_csv('sulphate_go0.06_curve.csv', index=False)
    print('Saved example to sulphate_go0.06_curve.csv')
