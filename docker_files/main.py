from sqlalchemy import create_engine
import os
from Prediction import Prediction
from flask import Flask, request, render_template


user = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASS')
host = os.getenv('POSTGRES_HOST')
port = os.getenv('POSTGRES_PORT')

print(user,password,host,port)
engine = create_engine(
    'postgresql+psycopg2://' + user + ':' + password + '@' + host + ':' + port + '/indiegogo_url')



app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/api/predict")
def Predict():
    q = request.args.get('q')
    prediction = Prediction(q, engine)
    return prediction.predict()


if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)

