import pyspark as ps
from pyspark.ml.feature import MinHashLSHModel
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.linalg import SparseVector
from pyspark.sql.functions import col

class Engine():
    def __init__(self):
        self.sc = ps.SparkContext(
            master='local[6]', # when running cluster-> spark://host:port
            appName='movieRecommender',
            pyFiles=[
                'movies.py',
                'app.py'
            ]
        )
        self.spark = ps.sql.SparkSession(self.sc)

        hash_model_path = 'models/minhash'
        self.hash_model = MinHashLSHModel.load(hash_model_path)

        als_model_path = 'models/als'
        self.als_model = ALSModel.load(als_model_path)

        user_ratings_path = 'data/sparse_matrix'
        user_ratings = self.sc.pickleFile(user_ratings_path)
        self.user_ratings_df = self.spark.createDataFrame(
            user_ratings,
            ['id', 'features']
        )

    def make_recommendations(self, id_ratings):
        id_ratings = {
            int(key): value for key, value in id_ratings.items()
        }
        user_inputs = SparseVector(17771, id_ratings)

        neighbor = self.jaccard_neighbor(user_inputs)
        if not neighbor:
            # return popular movies
            return None

        # create a df for model to make recommendations
        # doesn't work to init df with shape=1x1, use a placeholder to get passed error
        similair_user = self.sc.parallelize([(neighbor, '_')])

        similair_user_df = self.spark.createDataFrame(
            similair_user,
            ['user', '_']
        )\
        .select(
            'user'
        )

        recs = self.als_model.recommendForUserSubset(
            similair_user_df,
            10
        ).collect()

        movie_ids = [
            r['item'] for r in recs[0]['recommendations']
        ]
        return movie_ids

    def jaccard_neighbor(self, user_inputs):
        '''
        to get user_id:
        from pyspark.sql import functions as F
        user_id =\
        self.user_ratings_df.agg(F.max('id')).collect()[0]['max(id)']+1
        '''
        user_id = 2649429
        rec_user = self.sc.parallelize([(user_id, user_inputs)])
        rec_user_df = self.spark.createDataFrame(
            rec_user,
            ['id','features']
        )
        try:
            return self.hash_model.approxSimilarityJoin(
                self.user_ratings_df,
                rec_user_df,
                .6,
                distCol='JaccardDistance'
            )\
            .select(
                col('datasetA.id').alias('neighbor'),
                col('JaccardDistance')
            )\
            .orderBy(
                col('JaccardDistance')
            )\
            .first().neighbor

        except AttributeError:
            '''
            file_logger.warning('user didn't match with any training data')
            approxSimilarityJoin returns no result - no neighbor - sus
            '''
            return None
