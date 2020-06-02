import nltk
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from collections import Counter
import bs4 as bs
import pandas as pd

class Prediction:
    def __init__(self, q, rv):
        self.q = q
        self.rv = rv

    def predict(self):
        url = self.q
        LOAD_MORE_BUTTON_XPATH = '//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button'

        driver = webdriver.Chrome(ChromeDriverManager().install())
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

                counts_tot = counts_span+counts_div
                temp = pd.DataFrame.from_dict(counts_tot, orient='index').reset_index()
                temp_nltk = pd.DataFrame([temp[0]])
            except Exception as e:
                print(e)
                break





        lr = self.rv
        prediction = lr.transform(temp_nltk)
        if prediction ==1:
            print("The campaign will be successful!")
        else:
            print("The campaign will be unsuccessful :(")
