import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import bs4 as bs
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
            -perform a 10-fold cross-validation to optimize the model parameters
            The logistic regression model trained is then dumped into a .pickle file that will not require the model
            to be retrained every time the user wants to perform a prediction."""
            df = pd.read_sql_query('select * from "idf_ml_set_complete"', con=self.engine)
            df = df.fillna("0")
            stopwords_list = stopwords.words('english')
            vectorizer = TfidfVectorizer(analyzer='word',
                                         ngram_range=(1, 2),
                                         min_df=0.003,
                                         max_df=0.5,
                                         max_features=5000,
                                         stop_words=stopwords_list)

            url = self.q

            LOAD_MORE_BUTTON_XPATH = '//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button'
            try:

                driver = webdriver.Chrome(ChromeDriverManager().install())
                driver.get(url)
                loadMoreButton = driver.find_element_by_xpath(LOAD_MORE_BUTTON_XPATH)
                time.sleep(2)
                loadMoreButton.click()
                time.sleep(5)
                soup = bs.BeautifulSoup(driver.page_source, "html.parser")
                span_list = []
                for a in soup.find_all('span', {'class': "overviewSection-contentText"}):
                    span = a.text
                    span_list.append(span.lower())

                div_list = []
                for a in soup.find_all('div', {'class': "routerContentStory-storyBody"}):
                    div = a.text
                    div_list.append(div.lower())

                temp_dict = [
                    {
                        'lower_case_span': span_list,
                        'lower_case_div': div_list,
                        'collected_percentage': np.NaN
                    }
                ]





                temp = pd.DataFrame.from_records(temp_dict)
                df = pd.concat([df, temp])
                df['lower_case_span'] = df['lower_case_span'].apply(lambda x: " ".join(x))
                df['lower_case_div'] = df['lower_case_div'].apply(lambda x: " ".join(x))
                tfidf_matrix = vectorizer.fit_transform(df['lower_case_span'] + " " + df['lower_case_div'])
                cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                cosine_similarities_series = cosine_similarities[0]
                df = df.dropna()
                df['cosine_similarities'] = cosine_similarities_series
                df['collected_percentage_binary'] = [1 if x > 100 else 0 for x in
                                                     df['collected_percentage'].astype(float)]

                df_estimate = df[df['cosine_similarities'].astype(float) > 0.8]

                if df_estimate.empty:
                    print('Not enough data-points to evaluate')
                else:
                    sum_ones = df_estimate['collected_percentage_binary'].sum()
                    len_series = len(df_estimate['collected_percentage_binary'])
                    if sum_ones > len_series:
                        output = "The campaign will be unsuccessful :("
                    else:
                        output = "The campaign will be successful!!"
                    print("cos simil output:", output)

                sdf = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
                sdf['collected_percentage'] = df['collected_percentage']
                to_pred = sdf[sdf.isnull().any(1)]
                X_to_pred = to_pred.drop(columns=["collected_percentage"])

                sdf = sdf.dropna()

                sdf['collected_percentage_binary'] = [1 if x > 100 else 0 for x in
                                                      df['collected_percentage'].astype(float)]
                X = sdf.drop(columns=["collected_percentage", "collected_percentage_binary"])
                clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, sdf['collected_percentage_binary'])

                res = clf.predict(X_to_pred)
                if res[0] == 0:
                    output = "The campaign will be unsuccessful :("
                else:
                    output = "The campaign will be successful!!"
                print("log reg output:", output)


            except Exception as e:
                print(e)