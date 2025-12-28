"""Hybrid recommendation system combining content and collaborative filtering."""

from pyspark.sql.functions import col, min, max, when, row_number
from pyspark.sql.window import Window

from . import config


class HybridRecommender:
    """Hybrid recommender combining content-based and collaborative filtering."""
    
    def __init__(self, alpha=config.HYBRID_ALPHA):
        """Initialize hybrid recommender.
        
        Args:
            alpha: Weight for ALS (collaborative). Content-based weight is (1-alpha)
        """
        self.alpha = alpha
        
    def normalize_scores(self, df, score_col, normalized_col):
        """Normalize scores per user to [0, 1] range.
        
        Args:
            df: DataFrame with scores
            score_col: Column name with original scores
            normalized_col: Column name for normalized scores
            
        Returns:
            DataFrame with normalized scores
        """
        window = Window.partitionBy("user_id")
        
        normalized = (
            df
            .withColumn("min_score", min(score_col).over(window))
            .withColumn("max_score", max(score_col).over(window))
            .withColumn(
                normalized_col,
                when(col("max_score") > col("min_score"),
                     (col(score_col) - col("min_score")) / (col("max_score") - col("min_score"))
                ).otherwise(0.5)
            )
            .drop("min_score", "max_score")
        )
        
        return normalized
    
    def fuse(self, content_recs, als_recs, top_k=config.TOP_K):
        """Fuse content-based and collaborative filtering recommendations.
        
        Args:
            content_recs: DataFrame with user_id, item_id, and score/distance
            als_recs: DataFrame with user_id, item_id, als_score
            top_k: Number of final recommendations per user
            
        Returns:
            DataFrame with user_id, item_id, hybrid_score, rank
        """
        # Normalize content-based scores (convert distance to similarity score)
        # Lower distance = higher similarity, so we invert it
        content_normalized = self.normalize_scores(
            content_recs.withColumn("score", 1 / (col("distance") + 1)),
            "score",
            "content_score_norm"
        ).select("user_id", "item_id", "content_score_norm")
        
        # Normalize ALS scores
        als_normalized = self.normalize_scores(
            als_recs,
            "als_score",
            "als_score_norm"
        ).select("user_id", "item_id", "als_score_norm")
        
        # Combine both recommendation systems with weighted fusion
        hybrid_recs = (
            content_normalized
            .join(als_normalized, on=["user_id", "item_id"], how="full_outer")
            .fillna(0.0, subset=["content_score_norm", "als_score_norm"])
            .withColumn(
                "hybrid_score",
                self.alpha * col("als_score_norm") + (1 - self.alpha) * col("content_score_norm")
            )
        )
        
        # Rank by hybrid score
        window = Window.partitionBy("user_id").orderBy(col("hybrid_score").desc())
        
        ranked_hybrid = (
            hybrid_recs
            .withColumn("rank", row_number().over(window))
            .filter(col("rank") <= top_k)
        )
        
        return ranked_hybrid
