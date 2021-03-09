from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType

class Parser();
    def __init__(self, sc, spark):
        self.sc = sc
        self.spark = spark

        self.df = self.createDF()

    def createDF(self):
        schema = StructType(
            [
                StructField("user", IntegerType(), False),
        		StructField("rating", IntegerType(), False),
        		StructField("date", TimestampType(), True),
        		StructField("item", IntegerType(), False)
            ]
        )
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

            # when not on local cluster
            # save to hdfs - https://hdfs3.readthedocs.io/en/latest/api.html
            # os.system('hdfs dfs -put -f {} {}'.format(file_path, hdfsDest))

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
