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

        extractor = Extractor(engine, directory, 3000)
        extractor.extract()

        create_set = Create_set(engine)
        create_set.maskunion()
        #

model(extract =True)

url = 'https://www.indiegogo.com//projects/curated-wardrobes-versatile-responsibly-made-entrepreneurship-women'

ml_training = Prediction(url,engine)
ml_training.predict()



