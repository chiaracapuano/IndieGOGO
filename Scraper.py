import bs4 as bs
import urllib.request
import requests
import pickle
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer




class Scraper_Features:
    def __init__(self, engine):
        self.engine = engine



    def scrape_and_features(self):
        """Obtain list of movies URLs from Netflix movies home page,
        scrape each link (that corresponds to a movie) to obtain the description of the movie,
        identify tags from the description using Rake.
        The Netflix genre tags are also included in the tags.
        This class returns a df that contains Code (movie URL) and Tags (obtained from movie description+Netflix tags).
        Dump the identified tags in a pickled file.
        The pickled file will make the comparison between tags and word in input much faster."""
        con = self.engine.connect()
        df_urls = pd.read_sql_query('select * from "URLS"', con = self.engine)
        df_features = pd.DataFrame()
        LOAD_MORE_BUTTON_XPATH = '//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button'

        #//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button
        #//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button
        vec = TfidfVectorizer()
        stop_words = set(stopwords.words('english'))
        urls = []
        count = 0
        url_analysis=[]
        for url in df_urls["URL"]:
            count = count + 1
            print(count, url)

            if count < 5:

                driver = webdriver.Chrome(ChromeDriverManager().install())
                driver.get(url)
                urls.append(url)



                while True:
                    try:
                        text_analyze = []
                        loadMoreButton = driver.find_element_by_xpath(LOAD_MORE_BUTTON_XPATH)
                        time.sleep(2)
                        loadMoreButton.click()
                        time.sleep(5)
                        soup = bs.BeautifulSoup(driver.page_source, "html.parser")

                        for a in soup.find_all('span', {'class': "overviewSection-contentText"}):
                            for word in a.text.split():
                                if word not in stop_words and word.isalpha():
                                    word_lower = word.lower()
                                    text_analyze.append(word_lower)
                        for a in soup.find_all('div', {'class': "routerContentStory-storyBody"}):
                            for word in a.text.split():
                                if word not in stop_words and word.isalpha():
                                    word_lower = word.lower()
                                    text_analyze.append(word_lower)
                        print(text_analyze)

                        url_analysis.append(text_analyze)
                        print(url_analysis)
                        vec.fit(url_analysis)

                        df_features = pd.DataFrame(vec.transform(text_analyze).toarray(), columns=sorted(vec.vocabulary_.keys()))




                    except Exception as e:
                        print(e)
                        break
            else:
                break
        print(len(urls))
        print(df_features)
