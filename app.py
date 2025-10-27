from flask import Flask, render_template, request
from joblib import load
import numpy as np

#Load models & scaler
rf_model = load('RfModel.joblib')
lr_model = load('LReg.joblib')
scaler = load('Scaler.joblib')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            feature_names = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
            features = [float(request.form[x]) for x in feature_names]
            features = np.array(features).reshape(1, -1)

            #Scale and transform
            features_scaled = scaler.transform(features)
            features_rf = rf_model.apply(features_scaled)

            #Prediction
            pred = lr_model.predict(features_rf)[0]
            prediction = "Diabetic" if pred == 1 else "Not Diabetic"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)


