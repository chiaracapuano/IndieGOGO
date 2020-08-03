from bs4 import BeautifulSoup
import requests
import re
import json
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from jinja2 import Environment, FileSystemLoader
from sklearn.linear_model import LogisticRegressionCV
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class Prediction:
    def __init__(self, q, engine):
        self.q = q
        self.engine = engine

    def predict(self):
            """The function dumps the content of the Postgres table ml_set into a PySpark df.
            The df is manipulated in order to:
            -label the successful campaigns (>+100% funding) with 1, the others with 0
            -oversample the dataset to take care of the disparity in data labels (more 0s than 1s)
            -perform a 5-fold cross-validation to optimize the model parameters"""

            #Get the ad corpuses
            df = pd.read_sql_query('select * from "idf_ml_set_complete"', con=self.engine)
            df = df.fillna("0")
            #Get the user link to scrape+extract page json version
            url = self.q
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            json_data = soup.find('script', text=re.compile("//<!\[CDATA\["))
            pattern = ',"project_id":(.*)};gon.tracking_info={'
            project_id = re.search(pattern, str(json_data)).group(1)

            #Scrape page's json version
            page = requests.get("https://www.indiegogo.com/private_api/campaigns/" + project_id + "/description")
            soup = BeautifulSoup(page.content, 'html.parser')
            dict_json = json.loads(str(soup))
            html_str = dict_json['response']['description_html']
            soup_text = BeautifulSoup(html_str, "html.parser").get_text()

            #Append new page to pre-existing corpus
            temp_dict = [
                {
                    'lower_case_span': str(),
                    'lower_case_div': soup_text.lower(),
                    'collected_percentage': np.NaN
                }
            ]

            temp = pd.DataFrame.from_records(temp_dict)
            df_to_pred = pd.concat([df, temp])
            df_to_pred = df_to_pred.reset_index(drop=True)

            #Create TFIDF matrix of the whole corpus and transform it into a df
            stopwords_list = stopwords.words('english')
            vectorizer = TfidfVectorizer(analyzer='word',
                                         ngram_range=(1, 2),
                                         max_df=0.95,
                                         min_df=0.05,
                                         stop_words=stopwords_list)
            tfidf_matrix = vectorizer.fit_transform(df_to_pred['lower_case_span'] + " " + df_to_pred['lower_case_div'])
            sdf = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
            sdf['collected_percentage'] = df_to_pred['collected_percentage']

            #Dataframe to train the ML model
            sdf_train = sdf.dropna()

            #Create labels from collected_percentage
            sdf_train['collected_percentage'] = sdf_train['collected_percentage'].str.replace(",", ".")

            sdf_train['collected_percentage_binary'] = [1 if x > 100 else 0 for x in
                                                        sdf_train['collected_percentage'].astype(float)]

            X = sdf_train.drop(columns=["collected_percentage", "collected_percentage_binary"])
            y = sdf_train['collected_percentage_binary']

            #Train CV Logistic Regression
            clf = LogisticRegressionCV(cv=5, class_weight='balanced', max_iter=2000).fit(X, y)

            # Row of the matrix to be predicted
            X_to_pred = sdf[sdf.isnull().any(1)]
            X_to_pred = X_to_pred.drop(columns=["collected_percentage"])

            #Prediction
            res = clf.predict(X_to_pred)
            if res[0] == 0:
                output_logreg = "The campaign will be unsuccessful :("
            else:
                output_logreg = "The campaign will be successful!!"

            env = Environment(loader=FileSystemLoader('./templates'))

            template = env.get_template("output.html")

            template_vars = {"output_logreg": output_logreg}

            return template.render(template_vars)

















