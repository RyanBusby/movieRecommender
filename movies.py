from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col

class MovieDict():
    def __init__(self,  spark):
        schema = StructType(
            [
                StructField("id", IntegerType(), False),
                StructField("year", IntegerType(), False),
                StructField("name", StringType(), False)
            ]
        )
        filename = 'data/movie_titles.csv'
        self.movieTitles = spark.read.csv(filename, schema=schema)
        self.options = {
            f"{x['name']} - {x['year']}" : x['id']
            for x in self.movieTitles.collect()
        }

    def get_titles(self, movie_ids):
        return [
            f"{r.name} - {r.year}"
            for r in self.movieTitles\
                .filter(
                    col('id').isin(movie_ids)
                )\
                .select(
                    'name','year'
                )\
            .collect()
        ]
