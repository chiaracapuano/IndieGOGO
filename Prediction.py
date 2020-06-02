import nltk
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from collections import Counter
import bs4 as bs
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler


class Prediction:
    def __init__(self, q, loaded_model, sc):
        self.q = q
        self.loaded_model = loaded_model
        self.sc = sc

    def predict(self):
        url = self.q
        LOAD_MORE_BUTTON_XPATH = '//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button'

        driver = webdriver.Chrome(ChromeDriverManager().install())
        driver.get(url)
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
            temp_nltk.columns = temp['index']

            sqlCtx = SQLContext(self.sc)
            sdf = sqlCtx.createDataFrame(temp_nltk)

            features = sdf.schema.names

            assembler = VectorAssembler(inputCols=features, outputCol="Aspect")

            test = assembler.transform(sdf)
            prediction = self.loaded_model.transform(test)

            if prediction == 1:
                return "The campaign will be successful!"
            else:
                return "The campaign will be unsuccessful :("


        except Exception as e:
            print(e)



