# test_predictions.py
import pandas as pd
from hybrid_predictor import predict_single

# Embedded lab values (from your PDF)
lab_data = {
    "control": {
        0.00: {7: 26.53, 28: 33.33, 56: 47.14},
        0.02: {7: 31.29, 28: 35.37, 56: 41.54},
        0.04: {7: 32.65, 28: 38.09, 56: 42.12},
        0.06: {7: 36.05, 28: 42.21, 56: 52.13},
        0.08: {7: 33.33, 28: 36.05, 56: 46.14},
        0.10: {7: 31.29, 28: 33.33, 56: 44.13},
    },
    "sulphate": {
        0.00: {7: 23.9, 28: 30.0, 56: 32.2},
        0.02: {7: 28.2, 28: 31.8, 56: 35.0},
        0.04: {7: 29.4, 28: 34.3, 56: 37.1},
        0.06: {7: 32.4, 28: 38.0, 56: 40.9},
        0.08: {7: 30.0, 28: 32.4, 56: 35.0},
        0.10: {7: 28.2, 28: 30.0, 56: 31.9},
    },
    "acid": {
        0.00: {7: 30.5, 28: 35.8, 56: 32.2},
        0.02: {7: 31.2, 28: 36.5, 56: 35.0},
        0.04: {7: 32.0, 28: 37.1, 56: 36.0},
        0.06: {7: 32.8, 28: 38.0, 56: 37.5},
        0.08: {7: 31.5, 28: 36.0, 56: 34.0},
        0.10: {7: 30.0, 28: 34.5, 56: 32.0},
    },
    "chloride": {
        0.00: {7: 23.1, 28: 29.75, 56: 40.1},
        0.02: {7: 26.85, 28: 32.3, 56: 44.5},
        0.04: {7: 28.7, 28: 35.2, 56: 46.75},
        0.06: {7: 30.1, 28: 37.9, 56: 50.5},
        0.08: {7: 27.2, 28: 34.8, 56: 46.1},
        0.10: {7: 26.1, 28: 32.85, 56: 43.2},
    },
}

# Allowed error tolerance (MPa)
TOL = 1.5

def run_tests():
    results = []
    for attack, go_dict in lab_data.items():
        for go, dvals in go_dict.items():
            for day, lab_val in dvals.items():
                pred = predict_single(go, day, attack)
                err = abs(pred - lab_val)
                status = "PASS" if err <= TOL else "FAIL"
                print(f"{status} {attack} GO={go:.2f} day={day}: predicted {pred:.2f} vs lab {lab_val} (err={err:.2f})")
                results.append({
                    "attack": attack,
                    "GO_percent": go,
                    "day": day,
                    "lab_strength": lab_val,
                    "pred_strength": pred,
                    "error": err,
                    "status": status
                })
    df = pd.DataFrame(results)
    df.to_csv("prediction_validation.csv", index=False)
    print("\nValidation saved to prediction_validation.csv")

if __name__ == "__main__":
    run_tests()
