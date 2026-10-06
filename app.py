from flask import Flask, render_template, request
import numpy as np
import joblib


# -----------------------------------
# Create Flask application
# -----------------------------------

app = Flask(__name__)


# -----------------------------------
# Load trained ML model
# -----------------------------------

obj = joblib.load("winequality.joblib")

model = obj["model"]
cols = obj["columns"]


# -----------------------------------
# Home page
# -----------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None
    )


# -----------------------------------
# Prediction
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    values = [
        float(request.form["fixed_acidity"]),
        float(request.form["volatile_acidity"]),
        float(request.form["citric_acid"]),
        float(request.form["residual_sugar"]),
        float(request.form["chlorides"]),
        float(request.form["free_sulfur_dioxide"]),
        float(request.form["total_sulfur_dioxide"]),
        float(request.form["density"]),
        float(request.form["ph"]),
        float(request.form["sulphates"]),
        float(request.form["alcohol"])
    ]

    # Convert input into NumPy array
    input_data = np.array([values])

    # Prediction
    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=round(float(prediction), 2)
    )


# -----------------------------------
# Run application
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)