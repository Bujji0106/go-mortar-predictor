from flask import Flask, request, jsonify, render_template
from hybrid_predictor import predict_strength_curve, predict_weight_curve

app = Flask(__name__, template_folder='.')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/curve")
def curve():
    try:
        go = float(request.args.get("go", 0.06))
        attack = request.args.get("attack", "control").lower()
        days_max = int(request.args.get("days", 120))
    except Exception:
        return jsonify({"error": "Invalid input"}), 400

    # Predict curves
    days, strengths = predict_strength_curve(go, attack, days_max=days_max)
    _, weights = predict_weight_curve(go, attack, days_max=days_max)

    # Calculate decay info (from first to last point)
    if len(strengths) > 1:
        decay_rate = round((strengths[0] - strengths[-1]) / strengths[0] * 100, 2)
    else:
        decay_rate = 0.0

    return jsonify({
        "days": days.tolist(),
        "strengths": [float(x) for x in strengths],
        "weights": [float(x) for x in weights],
        "attack": attack,
        "decayRate": decay_rate
    })

if __name__ == "__main__":
    app.run(debug=True)
