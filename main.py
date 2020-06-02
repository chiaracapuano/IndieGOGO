from ML_training import ml_model
from sqlalchemy import create_engine
from Extractor import Extractor
from Scraper import Scraper_Features
import configparser
from flask import Flask, request, Response
import joblib

configParser = configparser.RawConfigParser()
configFilePath = './login.config'
configParser.read(configFilePath)
user = configParser.get('dev-postgres-config', 'user')
password = configParser.get('dev-postgres-config', 'pwd')
host = configParser.get('dev-postgres-config', 'host')
port = configParser.get('dev-postgres-config', 'port')
engine = create_engine('postgresql+psycopg2://'+user+':'+password+'@'+host+':'+port+'/indiegogo_url')
driver = "org.postgresql.Driver"
url = "jdbc:postgresql://"+host+":"+port+"/indiegogo_url"
table = "public.ml_set"

def model(extract = False):
    """If extract==True, then the files in RawFiles folder are scanned to append new data to the pre-existing feature
    database."""
    if extract ==True:
        directory = '/Users/chiara/PycharmProjects/IndieGOGO/RawFiles/'
        extractor = Extractor(engine, directory)
        extractor.extract()
        scraper_features = Scraper_Features
        scraper_features.scrape_and_features()
        ml_training = ml_model(user, password, host, port, driver, url, table)
        ml_training.make_model()

model(extract=True)
try:
    rv = joblib.load('update-database/filename.pickle')
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
                color:green; 
            } 
        </style>
    <body>
        <h1>Please enter a topic you'd like to watch a movie about</h1>
        <h3>Movies obtained from https://www.netflix.com/browse/genre/34399</h3>
        <div>
            <form action="/api/suggest" method="get">
                <label for="q">Topic:</label><br>
                <input type="text" id="q" name="q" value=""><br>
            </form>
        </div>
        </body>
    </html>
           """

@app.route("/api/suggest")
def Suggest():
    q = request.args.get('q')
    suggestions = Suggestions(df, df_ratings, q, rv, nlp)
    result = suggestions.calculate_weigths()
    return Response(result.to_json(orient="records"), mimetype='application/json')



if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000, debug=True, threaded = True, use_reloader=False)
