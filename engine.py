import pyspark as ps
from pyspark.ml.feature import MinHashLSHModel
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.linalg import SparseVector, Vectors

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
        '''
        TODO: since cosine (approxNearestNeighbor) does not take into account magnitude only direction, take that part out.
        Jaccarddistance max is .5 - change the parameter in the method call of the model here to .6
        also, sort the result to get the closest neighbor!
        '''


        id_ratings = {
            int(key): value for key, value in id_ratings.items()
        }
        user_inputs = Vectors.sparse(
            17771, id_ratings
        )

        neighbor = self.get_neighbor(user_inputs)
        if not neighbor:
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

    def cosine_neighbor(self, user_inputs):
        neighbors = self.hash_model.approxNearestNeighbors(
            self.user_ratings_df,
            user_inputs,
            1)\
        .collect()
        if len(neighbors) == 0:
            return None
        return neighbors[0].id

    def jaccard_neighbor(self, user_inputs, thresh):
        user_id = 2649429
        rec_user = self.sc.parallelize([(user_id, user_inputs)])
        rec_user_df = self.spark.createDataFrame(
            rec_user,
            ['id','features']
        )
        distances = self.hash_model.approxSimilarityJoin(
            self.user_ratings_df,
            rec_user_df,
            thresh
        )
        if len(distances.take(1)) == 0:
            return None
        return distances.filter(
            distances.distCol==distances.agg(
                F.min(distances.distCol)
            ).first()[0]
        )\
        .first()\
        .datasetA\
        .id

    def run_jaccard(self, user_inputs):
        for thresh in [1, 1.2, 1.4, 2, 3, 10]:
            neighbor = self.jaccard_neighbor(user_inputs, thresh)
            if neighbor:
                return neighbor
        return neighbor

    def get_neighbor(self, user_inputs):
        neighbor = self.cosine_neighbor(user_inputs)
        if not neighbor:
            neighbor = self.run_jaccard(user_inputs)
        return neighbor
