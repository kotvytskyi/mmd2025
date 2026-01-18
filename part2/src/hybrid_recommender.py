"""Hybrid recommendation system combining content and collaborative filtering."""

import time
from pyspark.sql.functions import col, min, max, when, row_number
from pyspark.sql.window import Window

from . import config


class HybridRecommender:
    def __init__(self, alpha=config.HYBRID_ALPHA):
        self.alpha = alpha
        
    def normalize_scores(self, df, score_col, normalized_col):
        stats = df.agg(
            min(score_col).alias("global_min"),
            max(score_col).alias("global_max")
        ).collect()[0]
        
        global_min = stats["global_min"]
        global_max = stats["global_max"]
        
        normalized = df.withColumn(
            normalized_col,
            (col(score_col) - global_min) / (global_max - global_min)
        )
        
        return normalized
    
    def fuse(self, content_recs, als_recs, top_k=config.TOP_K_PRECISION):
        start = time.time()
        
        content_normalized = self.normalize_scores(
            content_recs.withColumn("score", 1 / (col("distance") + 1)),  # lower distance - higher similarity
            "score",
            "content_score_norm"
        ).select("user_id", "item_id", "content_score_norm")
        
        als_normalized = self.normalize_scores(
            als_recs,
            "score",
            "als_score_norm"
        ).select("user_id", "item_id", "als_score_norm")
        
        hybrid_recs = (
            content_normalized
            .join(als_normalized, on=["user_id", "item_id"], how="full_outer")
            .fillna(0.0, subset=["content_score_norm", "als_score_norm"])
            .withColumn(
                "score",
                self.alpha * col("als_score_norm") + (1 - self.alpha) * col("content_score_norm")
            )
        )
        
        window = Window.partitionBy("user_id").orderBy(col("score").desc())
        
        ranked_hybrid = (
            hybrid_recs
            .withColumn("rank", row_number().over(window))
            .filter(col("rank") <= top_k)
        )
        
        ranked_hybrid.count()  # Force evaluation
        
        fuse_time = time.time() - start
        return ranked_hybrid, fuse_time
