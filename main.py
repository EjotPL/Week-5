from flask import Flask,url_for,request,render_template
import joblib
import pandas as pd


app = Flask(__name__)
model = joblib.load("/home/ejotpl91/MLApp/linear_regression.joblib")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv("/home/ejotpl91/MLApp/toy_data.csv")
    feature_data = data.values[:, :-1]
    predicted_data = model.predict(feature_data)
    return render_template('results.html', predicted_data=predicted_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)