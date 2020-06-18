import bs4 as bs
import pandas as pd
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


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



        df = pd.DataFrame()

        for url in df_urls["url"]:
            driver.get(url)

            collected_percentage = df_urls["collected_percentage"][count]
            count = count + 1

            while True:
                try:
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
                            'collected_percentage': collected_percentage[:-1]
                        }
                    ]
                    temp = pd.DataFrame.from_records(temp_dict)
                    df = pd.concat([df, temp])

                    date = self.date.replace("-", "_")
                    df.to_sql('TF-IDF_ml_set{}'.format(date), self.engine, if_exists='replace', index=False)




                except Exception as e:
                    print(e)
                    break










