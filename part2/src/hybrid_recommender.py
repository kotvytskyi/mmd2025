import time
from pyspark.sql.functions import col, coalesce, lit, row_number
from pyspark.sql.window import Window

from . import config


class HybridRecommender:
    def __init__(self, k=60):
        self.k = k
    
    def fuse(self, content_recs, als_recs, top_k=config.TOP_K_PRECISION):
        start = time.time()
        
        content_ranked = content_recs.select(
            col("user_id"), col("item_id"), col("rank").alias("content_rank")
        )
        
        als_window = Window.partitionBy("user_id").orderBy(col("score").desc())
        als_ranked = (
            als_recs
            .withColumn("als_rank", row_number().over(als_window))
            .select("user_id", "item_id", "als_rank")
        )
        
        default_rank = 999
        
        hybrid_recs = (
            content_ranked
            .join(als_ranked, on=["user_id", "item_id"], how="full_outer")
            .withColumn("content_rank", coalesce(col("content_rank"), lit(default_rank)))
            .withColumn("als_rank", coalesce(col("als_rank"), lit(default_rank)))
            .withColumn(
                "score",
                1.0 / (self.k + col("content_rank")) + 1.0 / (self.k + col("als_rank"))
            )
        )
        
        window = Window.partitionBy("user_id").orderBy(col("score").desc())
        
        ranked_hybrid = (
            hybrid_recs
            .withColumn("rank", row_number().over(window))
            .select("user_id", "item_id", "score", "rank")
            .persist()
        )
        
        ranked_hybrid.count()
        fuse_time = time.time() - start

        return ranked_hybrid, fuse_time