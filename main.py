from ModelPrep.Create_set import Create_set
from ModelPrep.Extractor import Extractor
from sqlalchemy import create_engine
import configparser
from Prediction import Prediction
from flask import Flask, request, render_template


configParser = configparser.RawConfigParser()
configFilePath = './login.config'
configParser.read(configFilePath)
user = configParser.get('dev-postgres-config', 'user')
password = configParser.get('dev-postgres-config', 'pwd')
host = configParser.get('dev-postgres-config', 'host')
port = configParser.get('dev-postgres-config', 'port')

engine = create_engine(
    'postgresql+psycopg2://' + user + ':' + password + '@' + host + ':' + port + '/indiegogo_url')


def model_prep(extract=False):
    """If extract==True, then the files in RawFiles/New_Files/ folder are scanned to append new data to the pre-existing
    corpus database."""
    if extract ==True:

        directory = '/Users/chiara/PycharmProjects/IndieGOGO/RawFiles/New_Files/'

        extractor = Extractor(engine, directory, 3000)
        extractor.extract()

        create_set = Create_set(engine)
        create_set.maskunion()


model_prep(extract=False)

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

    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True, use_reloader=False)

