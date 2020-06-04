from ModelPrep.Extractor import Extractor
from ModelPrep.ML_training import ml_model
from sqlalchemy import create_engine
import configparser
from ModelPrep.Scraper import Scraper_Features
from Prediction import Prediction
from flask import Flask, request
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
        driver = "org.postgresql.Driver"
        url = "jdbc:postgresql://" + host + ":" + port + "/indiegogo_url"
        table = "public.ml_set"
        directory = '/Users/chiara/PycharmProjects/IndieGOGO/RawFiles/'
        extractor = Extractor(engine, directory)
        extractor.extract()
        scraper_features = Scraper_Features(engine)
        scraper_features.scrape_and_features()
        ml_training = ml_model(user, password, host, port, driver, url, table, sc)
        ml_training.make_model()
#
model()
try:
    loaded_model = CrossValidatorModel.load("/Users/chiara/PycharmProjects/IndieGOGO/PySpark-cvLR-model")
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise
print("loaded file")


app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html><head></head>
    <style> 
            body { 
                text-align:center; 
            } 
            h1 { 
                color:black; 
            } 
        </style>
    <body>
        <h1>Please enter a IndieGOGO website link...</h1>
        <h3>to evaluate if the campaign will be successful</h3>
        <div>
            <form action="/api/suggest" method="get">
                <label for="q">Link:</label><br>
                <input type="text" id="q" name="q" value=""><br>
            </form>
        </div>
        </body>
    </html>
           """
@app.route("/api/suggest")
def Suggest():
    q = request.args.get('q')
    prediction = Prediction(q, loaded_model, sc)
    return prediction.predict()



if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000, debug=True, threaded = True, use_reloader=False)
