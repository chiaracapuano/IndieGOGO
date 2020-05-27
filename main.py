from Extractor import Extractor
from Scraper import Scraper_Features
from sqlalchemy import create_engine
import psycopg2
import configparser

configParser = configparser.RawConfigParser()
configFilePath = './login.config'
configParser.read(configFilePath)
user = configParser.get('dev-postgres-config', 'user')
pwd = configParser.get('dev-postgres-config', 'pwd')
host = configParser.get('dev-postgres-config', 'host')
port = configParser.get('dev-postgres-config', 'port')

engine = create_engine('postgresql+psycopg2://'+user+':'+pwd+'@'+host+':'+port+'/indiegogo_url')
#extractor = Extractor(engine)
#extractor.extract()
scraper_features = Scraper_Features(engine)
scraper_features.scrape_and_features()
