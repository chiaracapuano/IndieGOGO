# credits https://medium.com/@dhiraj.p.rai/logistic-regression-in-spark-ml-8a95b5f5434c
#https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
from pyspark.ml.feature import VectorAssembler, VectorSlicer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import *
from pyspark.ml.classification import  RandomForestClassifier
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd

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
    -identify through random forest the most relevant features
    -oversample the dataset to take care of the disparity in data labels (more 0s than 1s)
    -perform a 10-fold cross-validation to optimize the model parameters
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

    # convert to vector representation for MLlib
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    raw_data = assembler.transform(df)



    #identifying most relevant features
    rf = RandomForestClassifier(labelCol="collected_percentage_binary", featuresCol="features", seed=8464,
                                numTrees=10, cacheNodeIds=True, subsamplingRate=0.7)

    mod = rf.fit(raw_data)
    raw_data = mod.transform(raw_data)

    def ExtractFeatureImp(featureImp, dataset, featuresCol):
      list_extract = []
      for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
      varlist = pd.DataFrame(list_extract)
      varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
      return (varlist.sort_values('score', ascending=False))

    varlist = ExtractFeatureImp(mod.featureImportances, raw_data, "features")
    varidx = [x for x in varlist['idx'][0:10]]

    slicer = VectorSlicer(inputCol="features", outputCol="Sliced_features", indices=varidx)
    raw_data = slicer.transform(raw_data)
    raw_data = raw_data.drop('rawPrediction', 'probability', 'prediction')

    df_features = raw_data.drop('collected_percentage_binary')
    sliced_features = df_features.schema.names

    # convert to vector representation for MLlib

    assembler = VectorAssembler(inputCols=sliced_features, outputCol="Assembled_sliced_features")
    raw_data = assembler.transform(raw_data)


    # oversample to compensate for the disparity in data labels

    zeroes = raw_data.filter(col("collected_percentage_binary") == 0)
    ones = raw_data.filter(col("collected_percentage_binary") == 1)

    if zeroes.count() >ones.count():
      major_df = zeroes
      minor_df = ones
    else:
      major_df = ones
      minor_df = zeroes


    ratio = int(major_df.count() / minor_df.count())

    a = range(ratio)

    # duplicate the minority rows
    oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
    # combine both oversampled minority rows and previous majority rows
    combined_df = major_df.unionAll(oversampled_df)

    #random subdivision of dataset in test and training

    training, test = combined_df.randomSplit([0.8, 0.2], seed=12345)

    # logistic regression
    lr = LogisticRegression(labelCol="collected_percentage_binary", featuresCol="Assembled_sliced_features", maxIter=10)

    evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="collected_percentage_binary")


    paramGrid = ParamGridBuilder()\
        .addGrid(lr.aggregationDepth,[2,5,10])\
        .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
        .addGrid(lr.fitIntercept,[False, True])\
        .addGrid(lr.maxIter,[10, 100, 1000])\
        .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
        .build()

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(training)

    cvModel.write().overwrite().save("./PySpark-cvLR-model")

    print("Model Saved")