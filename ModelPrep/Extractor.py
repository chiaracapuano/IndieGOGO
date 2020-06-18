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
        con.execute('drop table if exists urls_old')
        con.execute('alter table if exists urls rename to urls_old')

        con.execute(
            'create table urls ( url varchar not null, category_name varchar, collected_percentage varchar)')

        for filename in os.listdir(directory):
            print(filename)
            if filename.endswith(".gz"):


                with gzip.open(directory+filename, 'rt') as gzip_file:
                    count = 0

                    data = gzip_file.read()
                    j = json.loads("[" +data.replace("}\n{", "},\n{") +"]")
                    date = re.search('Indiegogo(.*)T', filename)
                    df = pd.DataFrame()

                    for elem in j:
                        if count < 10:
                            url = 'https://www.indiegogo.com/'+elem["data"]["url"]
                            count = count + 1
                            print(url, count)

                            temp_dict = [
                                    {
                                        'url': url,
                                        'category_name': elem["data"]["category_name"],
                                        'collected_percentage': elem["data"]["collected_percentage"]
                                    }
                                ]
                            temp = pd.DataFrame.from_records(temp_dict)
                            df = pd.concat([df, temp])
                    df.to_sql('urls', self.engine, if_exists='replace', index = False)
                    print("scraping features now")
                    scraper_features = Scraper_Features(self.engine, date.group(1))
                    scraper_features.scrape_and_features()

            elif filename.endswith("json"):

                with open(directory+filename, 'r') as file:
                    count = 0

                    data = file.read()
                    j = json.loads("[" +data.replace("}\n{", "},\n{") +"]", encoding = "utf-8")
                    date = re.search('Indiegogo(.*)T', filename)
                    df = pd.DataFrame()

                    for elem in j:
                        if count < 10:
                            url = 'https://www.indiegogo.com/' + elem["data"]["url"]
                            count = count + 1
                            print(url, count)

                            temp_dict = [
                                {
                                    'url': url,
                                    'category_name': elem["data"]["category_name"],
                                    'collected_percentage': elem["data"]["collected_percentage"]
                                }
                            ]
                            temp = pd.DataFrame.from_records(temp_dict)
                            df = pd.concat([df, temp])
                    df.to_sql('urls', self.engine, if_exists='replace', index=False)
                    print("scraping features now")
                    scraper_features = Scraper_Features(self.engine, date.group(1))
                    scraper_features.scrape_and_features()
            print("URLS extracted")
