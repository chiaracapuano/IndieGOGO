# IndieGOGO
This repo contains the development of a Flask app, that evaluates whether a campaign launched on the IndieGOGO website will be successful or not.

The home page invites the user to input an IndieGOGO ad link:

![alt text](https://github.com/chiaracapuano/IndieGOGO/blob/master/png-examples/home-page.png)

a pre-trained PySpark logistic regression model will determine whether the campaign will be successful or not, based on the parts of speech extracted via NLP:

![alt text](https://github.com/chiaracapuano/IndieGOGO/blob/master/png-examples/output.png)


* DB creation/update
* Flask App developement

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

The folder **templates** contains the **home.html** file that renders the webpage when the Flask app is run.

**main.py** in the main folder contains the development of the Flask app. The app loads the previously trained ML model and allows the user to enter a link to an IndieGOGO campaign web address,
then **Prediction.py** scrapes the address counting the parts of speech in it. The Counter obtained is passed to the ML model which evaluates wheter the campaign will be successful or not.

Access to the Postgres DB is granted using the login details as per *login.file.example*.

NOTES: 
* interestingly, random forest can be used beforehand to identify the most relevant components of the DF (#https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
). However, in this case it did not make any difference.
* the dataset used for training uses data from May 2016. More recent campaigns have a different webpage structures so this app should be adapted to accommodate for more recent campaigns.


