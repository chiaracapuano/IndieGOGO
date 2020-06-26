import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
import joblib

class Model_training:
    def __init__(self, engine):
        self.engine = engine

    def train(self):

            """The function dumps the content of the Postgres table ml_set into a PySpark df.
            The df is manipulated in order to:
            -label the successful campaigns (>+100% funding) with 1, the others with 0
            -perform a 5-fold cross-validation to optimize the model parameters"""
            df = pd.read_sql_query('select * from "idf_ml_set_complete"', con=self.engine)
            df = df.fillna("0")
            stopwords_list = stopwords.words('english')
            vectorizer = TfidfVectorizer(analyzer='word',
                                         ngram_range=(1, 2),
                                         max_df=0.95,
                                         min_df=0.03,
                                         stop_words=stopwords_list)



            filename = '/Users/chiara/PycharmProjects/IndieGOGO/ModelPrep/vectorizer.sav'
            joblib.dump(vectorizer, filename)


            tfidf_matrix = vectorizer.fit_transform(df['lower_case_span'] + " " + df['lower_case_div'])

            sdf = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
            sdf['collected_percentage'] = df['collected_percentage']

            sdf['collected_percentage'] = sdf['collected_percentage'].str.replace(",", ".")
            sdf['collected_percentage_binary'] = [1 if x > 100 else 0 for x in
                                                  sdf['collected_percentage'].astype(float)]

            X = sdf.drop(columns=["collected_percentage", "collected_percentage_binary"])


            y = sdf['collected_percentage_binary']
            clf = LogisticRegressionCV(cv=5, class_weight= 'balanced', scoring='roc_auc', max_iter=1000).fit(X, y)

            filename = '/Users/chiara/PycharmProjects/IndieGOGO/ModelPrep/LogReg_rocauc.sav'

            joblib.dump(clf, filename)
            print('Finished with training. Score:', clf.score(X, y))



















