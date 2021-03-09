# Movie Recommender
### *Made scalable thanks to PySpark*

In this project, Spark's ALS matrix factorization is applied to the Netflix Prize data from the kaggle competition.
>spark.ml currently supports model-based collaborative filtering, in which users and products are described by a small set of latent factors that can be used to predict missing entries. spark.ml uses the alternating least squares (ALS) algorithm to learn these latent factors.

To overcome the cold-start problem ([read about that here](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html#cold-start-strategy)) the MinHashLSH class within Spark is utilized. MinHash is a technique for quickly estimating how similar two sets are. Oppose to re-training the whole model whenever a new user inputs their ratings, use MinHash to see which user that was present for training is closest to the new user. Get the recommendations for the closest user from the ALS model, and give them to the new user.


[Netfilx Prize Data](https://www.kaggle.com/netflix-inc/netflix-prize-data) • [ALS](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.recommendation.ALS.html?highlight=als#als) • [MinHash](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.ml.feature.MinHashLSH.html?highlight=minhash#minhashlsh)


### Flask App
The recommender is made accessible with a Flask web application. The Flask web application is made stylish with Bootstrap. Below are some screen shots of the application in action.

![recommendationWebApplication
](md_files/recommender_screen_shot.png)
