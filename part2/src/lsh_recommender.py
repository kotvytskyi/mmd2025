import time
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import BucketedRandomProjectionLSH

from . import config


class LSHRecommender:
    def __init__(
        self, 
        bucket_length=config.LSH_BUCKET_LENGTH,
        num_hash_tables=config.LSH_NUM_HASH_TABLES,
        threshold=config.LSH_THRESHOLD
    ):
        self.bucket_length = bucket_length
        self.num_hash_tables = num_hash_tables
        self.threshold = threshold
        self.lsh_model = None
        
    def fit(self, movies_profiles):
        start = time.time()
        
        lsh = BucketedRandomProjectionLSH(
            inputCol="features_norm",
            outputCol="hashes",
            bucketLength=self.bucket_length,
            numHashTables=self.num_hash_tables
        )
        
        self.lsh_model = lsh.fit(movies_profiles)
        
        fit_time = time.time() - start
        return self, fit_time
    
    def recommend(self, user_profiles, movies_profiles, train_ratings, top_k=config.TOP_K_RECALL):
        if self.lsh_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
                
        movies_lsh = movies_profiles.select(
            col("item_id"),
            col("features_norm")
        ).cache()
        
        users_lsh = user_profiles.select(
            col("user_id"),
            col("user_features_norm").alias("features_norm")
        ).cache()

        start = time.time()
        
        recommendations = self.lsh_model.approxSimilarityJoin(
            users_lsh,
            movies_lsh, 
            threshold=self.threshold,
            distCol="distance"
        ).select(
            col("datasetA.user_id").alias("user_id"),
            col("datasetB.item_id").alias("item_id"),
            col("distance")
        )

        recommendations.count()  # force eval
        
        recommend_time = time.time() - start
        
        already_rated = train_ratings.select("user_id", "item_id")
        recommendations = recommendations.join(
            already_rated,
            on=["user_id", "item_id"],
            how="left_anti"
        )
        
        window = Window.partitionBy("user_id").orderBy(col("distance").asc())
        
        ranked_recs = (
            recommendations
            .withColumn("rank", row_number().over(window))
            .filter(col("rank") <= top_k)
        )

        return ranked_recs, recommend_time
