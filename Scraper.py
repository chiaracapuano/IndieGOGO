import bs4 as bs
import pandas as pd
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from collections import Counter
import nltk
nltk.download('averaged_perceptron_tagger')


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

        count = 0
        driver = webdriver.Chrome(ChromeDriverManager().install())
        df_append = []
        for url in df_urls["URL"]:
            count = count + 1
            driver.get(url)
            print(url)
            collected_percentage = df_urls["collected_percentage"][count]
            print(collected_percentage)
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
                        tokens = nltk.word_tokenize(lower_case)
                        tags = nltk.pos_tag(tokens)
                        counts_span = Counter(tag for word, tag in tags if tag.isalpha())

                    for a in soup.find_all('div', {'class': "routerContentStory-storyBody"}):

                        div = a.text
                        lower_case = div.lower()
                        tokens = nltk.word_tokenize(lower_case)
                        tags = nltk.pos_tag(tokens)
                        counts_div = Counter(tag for word, tag in tags if tag.isalpha())

                    counts_tot =   counts_span+counts_div
                    temp = pd.DataFrame.from_dict(counts_tot, orient='index').reset_index()
                    temp_nltk = pd.DataFrame([temp[0]])
                    temp_nltk.columns = temp['index']
                    temp_nltk["collected_percentage"] = collected_percentage
                    temp_nltk["collected_percentage"] = temp_nltk["collected_percentage"].str[:-1]
                    df_append.append(temp_nltk)
                except Exception as e:
                    print(e)
                    break

            df_nltk = pd.concat(df_append)
            df_nltk.reset_index(inplace = True)
            df_nltk.drop(['index'], axis=1, inplace=True)


            df_nltk.to_sql('ML_SET', self.engine, if_exists='replace', index = False)
