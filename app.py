from flask import Flask, request, render_template_string
import nbformat
from nbconvert import PythonExporter
import numpy as np

app = Flask(__name__)

# Load and run the notebook once on startup
NOTEBOOK_PATH = "Decision_Tree.ipynb"
global_namespace = {}

def run_notebook_once():
    with open(NOTEBOOK_PATH) as f:
        nb = nbformat.read(f, as_version=4)
        exporter = PythonExporter()
        python_code, _ = exporter.from_notebook_node(nb)
        exec(python_code, global_namespace)

run_notebook_once()
clf = global_namespace["clf"]
iris = global_namespace["iris"]

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = ""
    user_inputs = [""] * 4  # to retain form input on reload

    if request.method == "POST":
        try:
            user_inputs = [request.form[f"feature{i}"] for i in range(4)]
            inputs = [float(x) for x in user_inputs]
            pred = clf.predict([inputs])[0]
            prediction = f"<div class='result'> Predicted Class: <b>{iris.target_names[pred]}</b></div>"
        except Exception as e:
            prediction = f"<div class='error'> Error: {str(e)}</div>"

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris Flower Predictor</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f0f8ff;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 600px;
                margin: 40px auto;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            label {
                display: block;
                margin-top: 10px;
                font-weight: bold;
            }
            input[type="text"] {
                width: 100%;
                padding: 8px;
                margin-top: 5px;
                border-radius: 6px;
                border: 1px solid #ccc;
                box-sizing: border-box;
            }
            input[type="submit"] {
                margin-top: 20px;
                width: 100%;
                background-color: #007BFF;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #0056b3;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background-color: #e6ffe6;
                color: #2e7d32;
                border: 1px solid #a5d6a7;
                border-radius: 8px;
                font-size: 18px;
                text-align: center;
            }
            .error {
                margin-top: 20px;
                padding: 15px;
                background-color: #ffe6e6;
                color: #c62828;
                border: 1px solid #ef9a9a;
                border-radius: 8px;
                font-size: 18px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Iris Flower Predictor</h1>
            <form method="post">
                {% for i in range(4) %}
                    <label>{{ feature_names[i] }}</label>
                    <input type="text" name="feature{{ i }}" value="{{ user_inputs[i] }}" required>
                {% endfor %}
                <input type="submit" value="Predict">
            </form>
            {{ prediction|safe }}
        </div>
    </body>
    </html>
    """, feature_names=iris.feature_names, prediction=prediction, user_inputs=user_inputs)

if __name__ == "__main__":
    app.run(debug=True)