from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
class ml_model:
  def __init__(self, engine):
    self.engine = engine


  def make_model(self):
    """The function dumps the content of the Postgres table ml_set into a PySpark df.
    The df is manipulated in order to:
    -label the successful campaigns (>+100% funding) with 1, the others with 0
    -oversample the dataset to take care of the disparity in data labels (more 0s than 1s)
    -perform a 10-fold cross-validation to optimize the model parameters
    The logistic regression model trained is then dumped into a .pickle file that will not require the model
    to be retrained every time the user wants to perform a prediction."""
    df = pd.read_sql_query('select * from "TF-IDF_ml_set_2016_05_14"', con=self.engine)
    df = df.fillna("")
    print(df)
    stopwords_list = stopwords.words('english')
    vectorizer = TfidfVectorizer(analyzer='word',
                                 ngram_range=(1, 2),
                                 min_df=0.003,
                                 max_df=0.5,
                                 max_features=5000,
                                 stop_words=stopwords_list)
    tfidf_matrix = vectorizer.fit_transform(df['lower_case_span'] + "" + df['lower_case_div'])

    tfidf_feature_names = vectorizer.get_feature_names()
    print(type(tfidf_feature_names))
