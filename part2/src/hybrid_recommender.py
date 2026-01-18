import time
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from . import config

class HybridRecommender:
    def __init__(self, k=60):
        self.k = k

    def fuse(self, content_recs, als_recs, top_k=config.TOP_K_RECALL):
        start = time.time()

        c = (content_recs
             .filter(F.col("rank") <= config.TOP_K_RECALL)
             .select(
                 "user_id", "item_id",
                 (F.lit(1.0) / (F.lit(self.k) + F.col("rank"))).alias("score")
             ))

        a = (als_recs
             .filter(F.col("rank") <= config.TOP_K_RECALL)
             .select(
                 "user_id", "item_id",
                 (F.lit(1.0) / (F.lit(self.k) + F.col("rank"))).alias("score")
             ))

        hybrid_recs = (c.unionByName(a)
                         .groupBy("user_id", "item_id")
                         .agg(F.sum("score").alias("score")))

        w = Window.partitionBy("user_id").orderBy(F.col("score").desc())
        ranked_hybrid = (hybrid_recs
                         .withColumn("rank", F.row_number().over(w))
                         .select("user_id", "item_id", "score", "rank")
                         .persist())

        ranked_hybrid.count()
        fuse_time = time.time() - start
        return ranked_hybrid, fuse_time