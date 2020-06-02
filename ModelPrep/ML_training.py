# credits https://medium.com/@dhiraj.p.rai/logistic-regression-in-spark-ml-8a95b5f5434c
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


class ml_model:
  def __init__(self, user, password, host, port, driver, url, table, sc):
    self.user = user
    self.password = password
    self.host = host
    self.port = port
    self.driver = driver
    self.url = url
    self.table = table
    self.sc = sc

  def make_model(self):
    """The function dumps the content of the Postgres table ml_set into a PySpark df.
    The df is manipulated in order to:
    -label the successful campaigns (>+100% funding) with 1, the others with 0
    -scale the features and add proper weigths for model training (the number of 1s is
    greater than the number of 0s)
    -perform a 5-fold cross-validation to optimize the model parameters
    The logistic regression model trained is then dumped into a .pickle file that will not require the model
    to be retrained every time the user wants to perform a prediction."""
    sqlContext = SQLContext(self.sc)



    df = sqlContext.read.format("jdbc")\
      .option("driver", self.driver)\
      .option("url", self.url)\
      .option("dbtable", self.table)\
      .option("user", self.user)\
      .option("password", self.password)\
      .load()

    df = df.fillna(0)
    df = df.withColumn('collected_percentage_binary',when(df.collected_percentage < 100,0).otherwise(1))
    df = df.drop('collected_percentage')
    df_features = df.drop('collected_percentage_binary')
    features = df_features.schema.names

    assembler = VectorAssembler(inputCols=features, outputCol="features")

    raw_data = assembler.transform(df)
    standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
    raw_data=standardscaler.fit(raw_data).transform(raw_data)

    training, test = raw_data.randomSplit([0.8, 0.2], seed=12345)

    dataset_size=float(training.select("collected_percentage_binary").count())
    numPositives=training.select("collected_percentage_binary").where('collected_percentage_binary == 1').count()
    numNegatives=float(dataset_size-numPositives)


    BalancingRatio= numNegatives/dataset_size

    training=training.withColumn("classWeights", when(training.collected_percentage_binary == 1,BalancingRatio).otherwise(1-BalancingRatio))



    css = ChiSqSelector(featuresCol='Scaled_features',outputCol='Aspect',labelCol='collected_percentage_binary',fpr=0.05)
    training=css.fit(training).transform(training)


    lr = LogisticRegression(labelCol="collected_percentage_binary", featuresCol="Aspect",weightCol="classWeights",maxIter=10)

    model = lr.fit(training)
    model.write().overwrite().save("./PySpark-LR-model")
    evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="collected_percentage_binary")

    paramGrid = ParamGridBuilder()\
        .addGrid(lr.aggregationDepth,[2,5,10])\
        .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
        .addGrid(lr.fitIntercept,[False, True])\
        .addGrid(lr.maxIter,[10, 100, 1000])\
        .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
        .build()

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    cvModel = cv.fit(training)

    cvModel.write().overwrite().save("./PySpark-cvLR-model")

    print("Model Saved")