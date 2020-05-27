import os
import json
import gzip
import pandas as pd
class Extractor:
    def __init__(self, engine):
        self.engine = engine



    def extract(self):

        directory = '/Users/chiara/PycharmProjects/IndieGOGO/RawFiles/'
        df = pd.DataFrame()
        con = self.engine.connect()

        con.execute('alter table if exists URLS rename to URLS_OLD')

        con.execute(
            'create table URLS ( URL varchar not null, category_name varchar, collected_percentage varchar)')

        for filename in os.listdir(directory):
            if filename.endswith(".gz"):

                with gzip.open(directory+filename, 'rt') as gzip_file:

                    data = gzip_file.read()
                    j = json.loads("[" +data.replace("}\n{", "},\n{") +"]")
                    for elem in j:
                        url = 'https://www.indiegogo.com/'+elem["data"]["url"]


                        temp_dict = [
                                {
                                    'URL': url,
                                    'category_name': elem["data"]["category_name"],
                                    'collected_percentage': elem["data"]["collected_percentage"]
                                }
                            ]
                        temp = pd.DataFrame.from_records(temp_dict)
                        temp.to_sql('URLS', self.engine, if_exists='append', index = False)

                        df = pd.concat([df, temp])

            elif filename.endswith("json"):
                with open(directory+filename, 'r') as file:

                    data = file.read()
                    j = json.loads("[" +data.replace("}\n{", "},\n{") +"]", encoding = "utf-8")
                    for elem in j:
                        url = 'https://www.indiegogo.com/' + elem["data"]["url"]

                        temp_dict = [
                            {
                                'URL': url,
                                'category_name': elem["data"]["category_name"],
                                'collected_percentage': elem["data"]["collected_percentage"]
                            }
                        ]
                        temp = pd.DataFrame.from_records(temp_dict)
                        temp.to_sql('URLS', self.engine, if_exists='append', index = False)

        return "URLS extracted"