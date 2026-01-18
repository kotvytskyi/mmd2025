import time
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS

from . import config


class ALSRecommender:
    def __init__(
        self,
        max_iter=config.ALS_MAX_ITER,
        reg_param=config.ALS_REG_PARAM,
        rank=config.ALS_RANK
    ):
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.als_model = None
        
    def fit(self, train_ratings):
        start = time.time()
        
        als = ALS(
            maxIter=self.max_iter,
            regParam=self.reg_param,
            userCol="user_id",
            itemCol="item_id",
            ratingCol="rating",
            coldStartStrategy="drop",
            rank=self.rank,
            nonnegative=True
        )
        
        self.als_model = als.fit(train_ratings)
        self.als_model.userFactors.count()  # Force evaluation
        
        fit_time = time.time() - start
        return self, fit_time
    
    def recommend(self, users, train_ratings, top_k=config.TOP_K_RECALL):
        if self.als_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        start = time.time()
        
        als_recs = self.als_model.recommendForUserSubset(users, top_k)
        als_recs.count()

        recommend_time = time.time() - start
        
        als_recs_flat = (
            als_recs
            .withColumn("rec", explode("recommendations"))
            .select(
                col("user_id"),
                col("rec.item_id").alias("item_id"),
                col("rec.rating").alias("score")
            )
        )
        
        already_rated = train_ratings.select("user_id", "item_id")
        als_recs_filtered = als_recs_flat.join(
            already_rated,
            on=["user_id", "item_id"],
            how="left_anti"
        )
        
        return als_recs_filtered, recommend_time
