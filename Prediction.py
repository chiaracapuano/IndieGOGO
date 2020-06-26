from bs4 import BeautifulSoup
import requests
import re
import json
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader
import joblib


class Prediction:
    def __init__(self, q, engine, n_features):
        self.q = q
        self.engine = engine
        self.n_features = n_features

    def predict(self):

            """The function dumps the content of the Postgres table ml_set into a PySpark df.
            The df is manipulated in order to:
            -label the successful campaigns (>+100% funding) with 1, the others with 0
            -oversample the dataset to take care of the disparity in data labels (more 0s than 1s)
            -perform a 5-fold cross-validation to optimize the model parameters"""
            df = pd.read_sql_query('select * from "idf_ml_set_complete"', con=self.engine)
            df = df.fillna("0")
            vectorizer = joblib.load('ModelPrep/vectorizer.sav')

            url = self.q

            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            json_data = soup.find('script', text=re.compile("//<!\[CDATA\["))

            pattern = ',"project_id":(.*)};gon.tracking_info={'
            project_id = re.search(pattern, str(json_data)).group(1)
            # identifying the required elements and obtaining it by sub string
            page = requests.get("https://www.indiegogo.com/private_api/campaigns/" + project_id + "/description")
            soup = BeautifulSoup(page.content, 'html.parser')
            dict_json = json.loads(str(soup))
            html_str = dict_json['response']['description_html']
            soup_text = BeautifulSoup(html_str, "html.parser").get_text()

            temp_dict = [
                {
                    'lower_case_span': str(),
                    'lower_case_div': soup_text.lower(),
                    'collected_percentage': np.NaN
                }
            ]

            temp = pd.DataFrame.from_records(temp_dict)
            df = pd.concat([df, temp])
            df = df.reset_index(drop=True)

            tfidf_matrix = vectorizer.fit_transform(df['lower_case_span'] + " " + df['lower_case_div'])



            sdf = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
            sdf['collected_percentage'] = df['collected_percentage']
            to_pred = sdf[sdf.isnull().any(1)]
            X_to_pred = to_pred.drop(columns=["collected_percentage"])

            X_to_pred = X_to_pred.iloc[:, : self.n_features]
            clf = joblib.load('ModelPrep/LogReg_rocauc.sav')


            res = clf.predict(X_to_pred)
            if res[0] == 0:
                output_logreg = "The campaign will be unsuccessful :("
            else:
                output_logreg = "The campaign will be successful!!"

            env = Environment(loader=FileSystemLoader('./templates'))

            template = env.get_template("output.html")

            template_vars = {"output_logreg": output_logreg}

            return template.render(template_vars)

















