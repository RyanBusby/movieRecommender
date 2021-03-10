
import pyspark as ps
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import MinHashLSH

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
# sc = ps.SparkContext(conf=conf)
sc = ps.SparkContext('local[6]')
spark = ps.sql.SparkSession(sc)

# parse raw data
parser = Parser(sc, spark, sample=True) # subset of data pre-parsed
df = parser.createDF()

# fit and save als
als = ALS(
    rank=8,
    seed=42,
    maxIter=10,
    regParam=0.1
)
als_model = als.fit(df)
als_model.save('../models/als_fromSample')

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

# what would rmse be when actually doing the minhash thing
'''

# save minhash
df = parser.make_sparse(df)
df.rdd.saveAsPickleFile('../data/sparse_matrix_fromSample')

mh = MinHashLSH()
mh.setInputCol('features')
mh.setOutputCol('hashes')

mh_model = mh.fit(df)
mh_model.save('../models/minhash_fromSample')
