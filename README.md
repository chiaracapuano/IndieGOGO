# IndieGOGO
This repo contains the development of a Flask app, that evaluates whether a campaign launched on the IndieGOGO website will be successful or not.
The app pulls data from a Postgres DB and evaluates the parts of speech that constitute the campaign ad. The parts of speech will be dumped in an array 
that a PySpark trained logistic regression algorithm will use to determine if the campaign will be successful or not.

* DB creation/update
* Flask App developement

The folder **ModelPrep** contains the classes:

* Extractor.py: which extracts the URLs of the campaigns from the json files offered by Web Robots at https://webrobots.io/indiegogo-dataset/,
and dumps them in a Postgres DB table.

* Scraper.py: which querys the URLs table produced by the previous class to get the campaigns web addresses. The URLs are scraped using Python
Selenium, and from each page a Counter and the amount of money (%) collected by the campaign is obtained. The Counter counts the parts of speech that 
consitute the campaign ad corpus (as per NLTK tokenizer). The Counters and relative money raised are appended to a Posgres table which will be used to train the ML algorithm.
* ML_training.py: the parts of speech collected in the Scraper are used to train a logistic regression algorithm in PySpark.
The labels used are extracted from the amount of money raised for the campaign (the label is 0 if the campaign raised less than 100% the goal, 
1 otherwise). 
The dataset is oversampled to compensate the higher count of 0s (almost 70% of the dataset). 
The algorithm is optimized performing a 10-fold cross validation.

**main.py** in the main folder contains the development of the Flask app. The app loads the previously trained ML model and allows the user to enter a link to an IndieGOGO campaign web address,
then **Prediction.py** scrapes the address counting the parts of speech in it. The Counter obtained is passed to the ML model which evaluates wheter the campaign will be successful or not.

Access to the Postgres DB is granted using the login details as per *login.file.example*.

NOTES: 
* interestingly, random forest can be used beforehand to identify the most relevant components of the DF (#https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
). However, in this case it did not make any difference.
* the dataset used for training uses data from May 2016. More recent campaigns have a different webpage structures so this app should be adapted to accommodate for more recent campaigns.

CONCLUSION: the model is still biased towards predicting 0s, resulting in a lot of false negatives despite the oversampling. One solution could be to increase the dataset size. Alternatively, it might be that the parts of speech that compose the ads are not a good indicator for the success of the campaign.
