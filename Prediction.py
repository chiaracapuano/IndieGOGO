from bs4 import BeautifulSoup
import requests
import re
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from jinja2 import Environment, FileSystemLoader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV

class Prediction:
    def __init__(self, q, engine):
        self.q = q
        self.engine = engine

    def predict(self):

            """The function dumps the content of the Postgres table ml_set into a PySpark df.
            The df is manipulated in order to:
            -label the successful campaigns (>+100% funding) with 1, the others with 0
            -oversample the dataset to take care of the disparity in data labels (more 0s than 1s)
            -perform a 5-fold cross-validation to optimize the model parameters
            The logistic regression model trained is then dumped into a .pickle file that will not require the model
            to be retrained every time the user wants to perform a prediction."""
            df = pd.read_sql_query('select * from "idf_ml_set_complete"', con=self.engine)
            df = df.fillna("0")
            stopwords_list = stopwords.words('english')
            vectorizer = TfidfVectorizer(analyzer='word',
                                         ngram_range=(1, 2),
                                         max_df=0.95,
                                         max_features=5000,
                                         stop_words=stopwords_list)

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
            print(soup_text)

            temp_dict = [
                {
                    'lower_case_span': [],
                    'lower_case_div': soup_text,
                    'collected_percentage': np.NaN
                }
            ]

            temp = pd.DataFrame.from_records(temp_dict)
            df = pd.concat([df, temp])

            df['lower_case_span'] = df['lower_case_span'].apply(lambda x: " ".join(x))
            #df['lower_case_div'] = df['lower_case_div'].apply(lambda x: " ".join(x))
            tfidf_matrix = vectorizer.fit_transform(df['lower_case_span'] + " " + df['lower_case_div'])
            cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            cosine_similarities_series = cosine_similarities[0]
            df = df.dropna()
            df['cosine_similarities'] = cosine_similarities_series
            df['collected_percentage'] = df['collected_percentage'].str.replace(",", ".")

            df['collected_percentage_binary'] = [1 if x > 100 else 0 for x in
                                                 df['collected_percentage'].astype(float)]

            df_estimate = df[df['cosine_similarities'].astype(float) > 0.8]

            if df_estimate.empty:
                output_cossim = 'Not enough data-points to evaluate'
            else:
                sum_ones = df_estimate['collected_percentage_binary'].sum()
                len_series = len(df_estimate['collected_percentage_binary'])
                if sum_ones > len_series:
                    output_cossim = "The campaign will be unsuccessful :("
                else:
                    output_cossim = "The campaign will be successful!!"
            sdf = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
            sdf['collected_percentage'] = df['collected_percentage']
            to_pred = sdf[sdf.isnull().any(1)]
            X_to_pred = to_pred.drop(columns=["collected_percentage"])

            sdf = sdf.dropna()
            sdf['collected_percentage'] = sdf['collected_percentage'].str.replace(",", ".")
            sdf['collected_percentage_binary'] = [1 if x > 100 else 0 for x in
                                                  sdf['collected_percentage'].astype(float)]
            ones_weight = (sdf[sdf['collected_percentage_binary'] == 1]).sum(1).sum()
            zeroes_weight = len(sdf['collected_percentage_binary']) - ones_weight
            print(ones_weight, zeroes_weight)
            X = sdf.drop(columns=["collected_percentage", "collected_percentage_binary"])

            w = {0: zeroes_weight, 1: ones_weight}

            y = sdf['collected_percentage_binary']
            clf = LogisticRegressionCV(cv=5, class_weight=w, scoring='roc_auc').fit(X, y)

            print(clf.score(X, y))

            res = clf.predict(X_to_pred)
            if res[0] == 0:
                output_logreg = "The campaign will be unsuccessful :("
            else:
                output_logreg = "The campaign will be successful!!"

            env = Environment(loader=FileSystemLoader('./templates'))

            template = env.get_template("output.html")

            template_vars = {"output_logreg": output_logreg}

            return template.render(template_vars)

















