import bs4 as bs
import pandas as pd
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
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
        df_urls = pd.read_sql_query('select * from "URLS"', con = self.engine)
        LOAD_MORE_BUTTON_XPATH = '//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button'

        vec = TfidfVectorizer()
        count = 0
        url_analysis=[]
        driver = webdriver.Chrome(ChromeDriverManager().install())
        collected_percentage = []
        for url in df_urls["URL"]:
            count = count + 1
            driver.get(url)

            collected_percentage.append(df_urls["collected_percentage"][count])
            while True:
                try:
                    loadMoreButton = driver.find_element_by_xpath(LOAD_MORE_BUTTON_XPATH)
                    time.sleep(2)
                    loadMoreButton.click()
                    time.sleep(5)
                    soup = bs.BeautifulSoup(driver.page_source, "html.parser")

                    for a in soup.find_all('span', {'class': "overviewSection-contentText"}):

                        span = a.text
                    for a in soup.find_all('div', {'class': "routerContentStory-storyBody"}):

                        div = a.text
                    url_analysis.append(span+' '+div)
                    vec.fit(url_analysis)
                    df_features = pd.DataFrame(vec.transform(url_analysis).toarray(),
                                               columns=sorted(vec.vocabulary_.keys()))
                    df_features["collected_percentage"] = collected_percentage
                    df_features["collected_percentage"] = df_features["collected_percentage"].str[:-1]
                    df_features.to_sql('ML_SET', self.engine, if_exists='replace')


                    print(df_features)




                except Exception as e:
                    print(e)
                    break


