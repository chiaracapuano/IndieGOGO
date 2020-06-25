# IndieGOGO
This repo contains the development of a Flask app, that evaluates whether a campaign launched on the IndieGOGO website will be successful or not.

The app workflow is illustrated below, and it takes into account that there are *two prediction algorithms tested (one per Github branch)*:

<kbd><img src="https://github.com/chiaracapuano/IndieGOGO/blob/TF-IDF/png-examples/Workflow.001.png" /></kbd>

The home page invites the user to input an IndieGOGO ad link:

<kbd><img src="https://github.com/chiaracapuano/IndieGOGO/blob/PySpark/png-examples/home-page.png" /></kbd>

A logistic regression model will determine whether the campaign will be successful or not, based on the parts of speech extracted via NLP:

<kbd><img src="https://github.com/chiaracapuano/IndieGOGO/blob/PySpark/png-examples/output.png" /></kbd>

The Python codebase is contained in the three folders of this repo:

* home folder
* ModelPrep
* templates

The ML model used n the **PySpark** branch is a PySpark logistic regression pre-trained mode, while in the **TD_IDF** branch the model is a sklearn trained logistic regression model.

#### home folder
The Flask app corpus is contained in the file **main.py**. The app loads the previously trained ML model in the folder **PySpark-cvLR-ml_set_complete** and allows the user to enter a link to an IndieGOGO campaign web address.

### PySpark Branch

**Prediction.py** is called in the main, it scrapes the address in input and evaluates the parts of speech in the ad, storing them in a Counter. The Counter obtained is passed to the ML model, which computes wheter the campaign will be successful or not.

### TF-IDF Branch

**Prediction.py** is called in the main. It dumps the previously scraped addresses that form the database, scrapes the address in input and creates a dataframe appending the two. A tf-idf matrix is created, and its features used to train a sklearn logistic regression model (the user input link is obviously not included in the training).

The success of the campaign in the link in input is then predicted. 


The main also contains a function *model* that defaults to Flase, which can be used to retrain the model as well as create a whole new training-test dataset.

#### ModelPrep

### Master Branch

The folder **ModelPrep** contains the classes:

* Extractor.py: which extracts the URLs of the campaigns from the json files offered by Web Robots at https://webrobots.io/indiegogo-dataset/ (saved in a specific local folder),
and dumps them in a Postgres DB table. In Extractor.py, Scraper.py querys the URLs table produced by the extractor and scrapes the URLs using Python
Selenium. From each page a Counter and the amount of money (%) collected by the campaign is obtained. The Counter counts the parts of speech that 
consitute the campaign ad corpus (as per NLTK tokenizer). The Counters and money raised are appended to a Posgres table which will be used to train the ML algorithm. A table is produced for each json file in the local folder containing the campaign URLs: IndieGOGO is scraped by webrobots every month, therefore the table name contains the same date as the json file it has been created from.
* Create_set.py: connects to the DB and runs a stored prcedure saved n the DB, which unions all the tables created in the previous step into a single table (ml_set_complete) to be used to train the ML algorithm.
* ML_training.py: the table created in the previous step is used to train a logistic regression algorithm in PySpark.
The labels used are extracted from the amount of money raised for the campaign (the label is 0 if the campaign raised less than 100% the goal, 
1 otherwise). Around 10000 data points used for training and testing of the logistic regression algorithm lead to an area under the ROC curve of about 60%, meaning that either further data points are needed or that the parts of speech of a campaign might not be an indicator of its success.
The dataset is oversampled to compensate the higher count of 0s (almost 70% of the dataset). 
The algorithm is optimized performing a 10-fold cross validation.



### TF-IDF Branch

The folder **ModelPrep** contains the classes:

* Extractor.py: which extracts the URLs of the campaigns from the json files offered by Web Robots at https://webrobots.io/indiegogo-dataset/ (saved in a specific local folder),
and dumps them in a Postgres DB table. In Extractor.py, Scraper.py querys the URLs table produced by the extractor and scrapes the URLs using Python
Selenium. Each url is scraped and the text dumped in a Postgres table. A table is produced for each json file in the local folder containing the campaign URLs: IndieGOGO is scraped by webrobots every month, therefore the table name contains the same date as the json file it has been created from.
* Create_set.py: connects to the DB and runs a stored prcedure saved n the DB, which unions all the tables created in the previous step into a single table (ml_set_complete) to be used to train the ML algorithm.

Access to the Postgres DB is granted using the login details as per *login.file.example*.


#### Templates

The folder **templates** contains:
* **home.html**: file that renders the home page of the Flask app
* **output.html**: file that renders the dataframe visualized in output as movie suggestions.



NOTES: 
* interestingly, random forest can be used beforehand to identify the most relevant components of the DF (https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
). However, in this case it did not make any difference.
* the dataset used for training uses data from May 2016. More recent campaigns have a different webpage structures so this app should be adapted to accommodate for more recent campaigns.


