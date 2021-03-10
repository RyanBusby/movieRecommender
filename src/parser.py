
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType
from pyspark.ml.linalg import SparseVector

class Parser():
    def __init__(self, sc, spark, sample=False):
        self.sc = sc
        self.spark = spark
        self.sample = sample

    def createDF(self):
        schema = StructType(
            [
                StructField("user", IntegerType(), False),
        		StructField("rating", IntegerType(), False),
        		StructField("date", TimestampType(), True),
        		StructField("item", IntegerType(), False)
            ]
        )
        if self.sample:
            df = self.spark.read.csv(
                '../data/sample.csv',
                schema=schema
            )
            return df
        fnames = [
            f'../data/netflix-prize-data/combined_data_{n}.txt'\
            for n in range(1,5)
        ]
        df = self.spark.createDataFrame(
            self.sc.parallelize(
                self.parse_text_file(fnames[0])
            ),
            schema=schema
        )
        for fname in fnames[1:]:

            # another possibility
            # https://hdfs3.readthedocs.io/en/latest/api.html
            # save munged portion on namenode (fname)
            # if first: os.system('hdfs dfs -put -f {} {}'.format(fname, hdfsDest))
            # else: os.system('hdfs dfs -appendToFile {} {}'.format(fname, hdfsDest))

            df = df.union(
                self.spark.createDataFrame(
                    self.sc.parallelize(
                        self.parse_text_file(fname)
                    ),
                    schema=schema
                )
            )
        return df

    def parse_text_file(self, fname):
        with open(fname, 'r') as f:
            rows = f.read().splitlines()
        movie_id = None
        to_spark = []
        for row in rows:
            if ':' in row:
                movie_id = row[:-1]
                continue
            # insert directly into hdfs or into hive table?
            to_spark.append(tuple(row.split(',')+[movie_id]))
        return to_spark

    def make_sparse(self, df):
        return self.spark.createDataFrame(
            df.rdd.map(
                lambda x: (x.user, [(x.item, x.rating)])
            )\
            .reduceByKey(
                lambda x, y: x + y
            )\
            .map(
                lambda x: (x[0], SparseVector(17771, x[1]))
            ),
            ['user', 'features']
        )
