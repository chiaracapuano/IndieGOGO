from ModelPrep.Create_set import Create_set
from ModelPrep.Extractor import Extractor
from ModelPrep.ML_training import ml_model
from sqlalchemy import create_engine
import configparser
from Prediction import Prediction
from flask import Flask, request, render_template
from pyspark.ml.tuning import CrossValidatorModel
import sys
from pyspark import SparkContext


sc = SparkContext(appName="prediction")



def model(extract = False):
    """If extract==True, then the files in RawFiles folder are scanned to append new data to the pre-existing feature
    database."""
    if extract ==True:
        configParser = configparser.RawConfigParser()
        configFilePath = './login.config'
        configParser.read(configFilePath)
        user = configParser.get('dev-postgres-config', 'user')
        password = configParser.get('dev-postgres-config', 'pwd')
        host = configParser.get('dev-postgres-config', 'host')
        port = configParser.get('dev-postgres-config', 'port')
        engine = create_engine(
            'postgresql+psycopg2://' + user + ':' + password + '@' + host + ':' + port + '/indiegogo_url')
        directory = '/Users/chiara/PycharmProjects/IndieGOGO/RawFiles/'

        extractor = Extractor(engine, directory)
        extractor.extract()

        create_set = Create_set(engine)
        create_set.maskunion()
        driver = "org.postgresql.Driver"
        url = "jdbc:postgresql://" + host + ":" + port + "/indiegogo_url"
        table = "public.ml_set_complete"

        ml_training = ml_model(user, password, host, port, driver, url, table, sc)
        ml_training.make_model()

model()




try:
    loaded_model = CrossValidatorModel.load("/Users/chiara/PycharmProjects/IndieGOGO/PySpark-cvLR-ml_set_complete")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
print("loaded file")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
@app.route("/api/suggest")
def Suggest():
    q = request.args.get('q')
    prediction = Prediction(q, loaded_model, sc)
    return prediction.predict()



if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000, debug=True, threaded = True, use_reloader=False)
