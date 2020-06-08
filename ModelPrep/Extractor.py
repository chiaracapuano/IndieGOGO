import os
import json
import gzip
import pandas as pd
import re

from ModelPrep.Scraper import Scraper_Features


class Extractor:
    def __init__(self, engine, directory):
        self.engine = engine
        self.directory = directory



    def extract(self):
        """This function extracts the projects URLS in IndieGOGO from the json files in the RawFiles folder
        and dumps them in a Postgres table."""

        directory = self.directory
        con = self.engine.connect()

        con.execute('alter table if exists urls rename to urls_old')

        con.execute(
            'create table urls ( url varchar not null, category_name varchar, collected_percentage varchar)')

        for filename in os.listdir(directory):
            print(filename)
            if filename.endswith(".gz"):


                with gzip.open(directory+filename, 'rt') as gzip_file:

                    data = gzip_file.read()
                    j = json.loads("[" +data.replace("}\n{", "},\n{") +"]")
                    date = re.search('Indiegogo(.*)T', filename)
                    for elem in j:
                        url = 'https://www.indiegogo.com/'+elem["data"]["url"]


                        temp_dict = [
                                {
                                    'url': url,
                                    'category_name': elem["data"]["category_name"],
                                    'collected_percentage': elem["data"]["collected_percentage"],
                                }
                            ]
                        temp = pd.DataFrame.from_records(temp_dict)
                        temp.to_sql('urls', self.engine, if_exists='replace', index = False)
                    scraper_features = Scraper_Features(self.engine, date)
                    scraper_features.scrape_and_features()

            elif filename.endswith("json"):

                with open(directory+filename, 'r') as file:

                    data = file.read()
                    j = json.loads("[" +data.replace("}\n{", "},\n{") +"]", encoding = "utf-8")
                    date = re.search('Indiegogo(.*)T', filename)
                    for elem in j:
                        url = 'https://www.indiegogo.com/' + elem["data"]["url"]

                        temp_dict = [
                            {
                                'url': url,
                                'category_name': elem["data"]["category_name"],
                                'collected_percentage': elem["data"]["collected_percentage"],

                            }
                        ]
                        temp = pd.DataFrame.from_records(temp_dict)
                        temp.to_sql('urls', self.engine, if_exists='replace', index = False)
                    scraper_features = Scraper_Features(self.engine, date)
                    scraper_features.scrape_and_features()
        print("URLS extracted")
