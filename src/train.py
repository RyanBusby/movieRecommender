
import pyspark as ps
from pyspark.ml.recommendation import ALS

from parser import Parser

# init spark context
conf = ps.SparkConf().setAll(
    [
        ('spark.executor.memory', '8g'),
        ('spark.executor.cores', '8'),
        ('spark.driver.memory', '8g')
    ]
)
# specific to my machine, juiced up memory to train.
# https://spark.apache.org/docs/3.1.1/configuration.html#available-properties
sc = ps.SparkContext(conf=conf)
spark = ps.sql.SparkSession(sc)

# parse raw data
parser = Parser(sc, spark)
df = parser.createDF()

# fit and save als
als = ALS(
    rank=8,
    seed=42,
    maxIter=10,
    regParam=0.1
)
model = als.fit(df)
model.save('../models/als_')

# cross validate
'''
from pyspark.ml.evaluation import RegressionEvaluator

(training, test) = df.randomSplit([0.8, 0.2])

als = ALS(
    rank=8,
    seed=42,
    maxIter=10,
    regParam=0.01,
    coldStartStrategy="drop"
)
model = als.fit(training)

predictions = model.transform(test)
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
model.save('../models/als_%s' % rmse)
'''
