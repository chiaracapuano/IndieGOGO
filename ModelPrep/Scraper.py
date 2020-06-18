import bs4 as bs
import pandas as pd
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('averaged_perceptron_tagger')


class Scraper_Features:
    def __init__(self, engine, date):
        self.engine = engine
        self.date = date



    def scrape_and_features(self):
        """This function scrapes the URLs listed in the Postgres table, filled by the Extractpr.
        Selenium is used to push the "LOAD MORE" button, in order to scrape the full webpage.
        TF-IDF is used to convert the text into a vector structure.
        The results are dumped in the ml_set Postgres table, and will be later used to train the logistic regression model."""

        df_urls = pd.read_sql_query('select * from "urls"', con = self.engine)
        LOAD_MORE_BUTTON_XPATH = '//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button'

        count = 0
        driver = webdriver.Chrome(ChromeDriverManager().install())
        df_append = []
        stopwords_list = stopwords.words('english')
        vectorizer = TfidfVectorizer(analyzer='word',
                                     ngram_range=(1, 2),
                                     min_df=0.003,
                                     max_df=0.5,
                                     max_features=5000,
                                     stop_words=stopwords_list)



        for url in df_urls["url"]:
            counts_tot_list=[]
            count = count + 1
            driver.get(url)
            collected_percentage = df_urls["collected_percentage"][count]
            while True:
                try:
                    loadMoreButton = driver.find_element_by_xpath(LOAD_MORE_BUTTON_XPATH)
                    time.sleep(2)
                    loadMoreButton.click()
                    time.sleep(5)
                    soup = bs.BeautifulSoup(driver.page_source, "html.parser")

                    for a in soup.find_all('span', {'class': "overviewSection-contentText"}):
                        span = a.text
                        lower_case = span.lower()

                    for a in soup.find_all('div', {'class': "routerContentStory-storyBody"}):
                        div = a.text
                        lower_case = div.lower()

                    tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])

                    counts_tot = Counter()
                    for x in counts_tot_list:
                        counts_tot += x

                    for feature in model_features:
                        if feature not in counts_tot.keys():
                            counts_tot[feature] = 0

                    temp = pd.DataFrame.from_dict(counts_tot, orient='index').reset_index()
                    temp_nltk = pd.DataFrame([temp[0]])
                    temp_nltk.columns = temp['index']
                    temp_nltk["COLLECTED_PERCENTAGE"] = collected_percentage
                    temp_nltk["COLLECTED_PERCENTAGE"] = temp_nltk["COLLECTED_PERCENTAGE"].str[:-1]
                    df_append.append(temp_nltk)
                except Exception as e:
                    print(e)
                    break

                df_nltk = pd.concat(df_append)
                df_nltk.reset_index(inplace = True)
                df_nltk.drop(['index'], axis=1, inplace=True)
                df_nltk = df_nltk.reindex(sorted(df_nltk.columns), axis=1)

                date = self.date.replace("-","_")
                df_nltk.to_sql('sorted_ml_set{}'.format(date), self.engine, if_exists='replace', index = False)



