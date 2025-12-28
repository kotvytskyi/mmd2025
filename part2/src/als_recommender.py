"""ALS-based collaborative filtering recommendation system."""

from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS

from . import config


class ALSRecommender:
    """ALS-based collaborative filtering recommender."""
    
    def __init__(
        self,
        max_iter=config.ALS_MAX_ITER,
        reg_param=config.ALS_REG_PARAM,
        rank=config.ALS_RANK
    ):
        """Initialize ALS recommender.
        
        Args:
            max_iter: Maximum number of iterations
            reg_param: Regularization parameter
            rank: Number of latent factors
        """
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.rank = rank
        self.als_model = None
        
    def fit(self, train_ratings):
        """Fit ALS model on training ratings.
        
        Args:
            train_ratings: DataFrame with user_id, item_id, rating
        """
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
        return self
    
    def recommend(self, users, train_ratings, top_k=config.TOP_K):
        """Generate recommendations using ALS.
        
        Args:
            users: DataFrame with user_id (users to recommend for)
            train_ratings: DataFrame with already rated items to exclude
            top_k: Number of recommendations per user
            
        Returns:
            DataFrame with user_id, item_id, als_score
        """
        if self.als_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate recommendations
        als_recs = self.als_model.recommendForUserSubset(users, top_k)
        
        # Flatten recommendations
        als_recs_flat = (
            als_recs
            .withColumn("rec", explode("recommendations"))
            .select(
                col("user_id"),
                col("rec.item_id").alias("item_id"),
                col("rec.rating").alias("als_score")
            )
        )
        
        # Filter out already rated items
        already_rated = train_ratings.select("user_id", "item_id")
        als_recs_flat = als_recs_flat.join(
            already_rated,
            on=["user_id", "item_id"],
            how="left_anti"
        )
        
        return als_recs_flat
