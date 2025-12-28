"""LSH-based content recommendation system."""

from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark import StorageLevel

from . import config


class LSHRecommender:
    """LSH-based recommender using content features."""
    
    def __init__(
        self, 
        bucket_length=config.LSH_BUCKET_LENGTH,
        num_hash_tables=config.LSH_NUM_HASH_TABLES,
        threshold=config.LSH_THRESHOLD
    ):
        """Initialize LSH recommender.
        
        Args:
            bucket_length: LSH bucket length parameter
            num_hash_tables: Number of hash tables for LSH
            threshold: Distance threshold for similarity join
        """
        self.bucket_length = bucket_length
        self.num_hash_tables = num_hash_tables
        self.threshold = threshold
        self.lsh_model = None
        
    def fit(self, movies_profiles):
        """Fit LSH model on movie profiles.
        
        Args:
            movies_profiles: DataFrame with item_id and features_norm
        """
        lsh = BucketedRandomProjectionLSH(
            inputCol="features_norm",
            outputCol="hashes",
            bucketLength=self.bucket_length,
            numHashTables=self.num_hash_tables
        )
        
        self.lsh_model = lsh.fit(movies_profiles)
        return self
    
    def recommend(self, user_profiles, movies_profiles, train_ratings, top_k=config.TOP_K):
        """Generate recommendations using LSH.
        
        Args:
            user_profiles: DataFrame with user_id and user_features_norm
            movies_profiles: DataFrame with item_id and features_norm
            train_ratings: DataFrame with already rated items to exclude
            top_k: Number of recommendations per user
            
        Returns:
            DataFrame with user_id, item_id, distance, rank
        """
        if self.lsh_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare data for LSH
        movies_lsh = movies_profiles.select(
            col("item_id"),
            col("features_norm")
        ).cache()
        
        users_lsh = user_profiles.select(
            col("user_id"),
            col("user_features_norm").alias("features_norm")
        ).cache()
        
        # Find similar items for each user
        recommendations = self.lsh_model.approxSimilarityJoin(
            users_lsh,
            movies_lsh, 
            threshold=self.threshold,
            distCol="distance"
        ).select(
            col("datasetA.user_id").alias("user_id"),
            col("datasetB.item_id").alias("item_id"),
            col("distance")
        ).persist(StorageLevel.MEMORY_AND_DISK)
        
        # Filter out already rated items
        already_rated = train_ratings.select("user_id", "item_id")
        recommendations = recommendations.join(
            already_rated,
            on=["user_id", "item_id"],
            how="left_anti"
        )
        
        # Rank by distance (lower is better)
        window = Window.partitionBy("user_id").orderBy(col("distance").asc())
        
        ranked_recs = (
            recommendations
            .withColumn("rank", row_number().over(window))
            .filter(col("rank") <= top_k)
        )
        
        return ranked_recs
