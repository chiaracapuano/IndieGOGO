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
        model_features = []
        with open("/Users/chiara/PycharmProjects/IndieGOGO/ModelPrep/featurenames.txt") as file:
            for line in file:
                line = line.strip()  # or some other preprocessing
                model_features.append(line)  # storing everything in memory!
        url = self.q
        counts_tot_list=[]
        LOAD_MORE_BUTTON_XPATH = '//*[@id="vCampaignRouterContent"]/div[2]/div/div[2]/button'
        try:

            driver = webdriver.Chrome(ChromeDriverManager().install())
            driver.get(url)
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
                counts_tot_list.append(counts_span)
            for a in soup.find_all('div', {'class': "routerContentStory-storyBody"}):
                div = a.text
                lower_case = div.lower()
                tokens = nltk.word_tokenize(lower_case)
                tags = nltk.pos_tag(tokens)
                counts_div = Counter(tag for word, tag in tags if tag.isalpha())
                counts_tot_list.append(counts_div)

            counts_tot = Counter()
            for x in counts_tot_list:
                counts_tot += x
            print(counts_tot)

            for el in model_features:
                if el not in counts_tot.keys():
                    counts_tot[el] = 0
                else:
                    continue
            print(counts_tot)

            temp = pd.DataFrame.from_dict(counts_tot, orient='index').reset_index()
            temp_nltk = pd.DataFrame([temp[0]])
            temp_nltk.columns = temp['index']

            sqlCtx = SQLContext(self.sc)
            sdf = sqlCtx.createDataFrame(temp_nltk)

            features = sdf.schema.names

            assembler = VectorAssembler(inputCols=features, outputCol="features")

            test = assembler.transform(sdf)

            prediction = self.loaded_model.transform(test)

            result = prediction.select("prediction").toPandas()

            if result["prediction"][0] == 0:
                return "The campaign will be unsuccessful :("
            else:
                return "The campaign will be successful!!"




        except Exception as e:
            print(e)



