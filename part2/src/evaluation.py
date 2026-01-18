"""Evaluation metrics for recommendation systems."""

from pyspark.sql.functions import col, collect_list, struct, avg, count, min as spark_min, max as spark_max, row_number
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics

from . import config


def evaluate_recommendations(recommendations, test_ratings, 
                            precision_k=config.TOP_K_PRECISION, 
                            recall_k=config.TOP_K_RECALL,
                            ndcg_k=config.TOP_K_NDCG,
                            relevance_threshold=config.RELEVANCE_THRESHOLD):
    max_k = max(precision_k, recall_k, ndcg_k)

    users_with_enough_test_items = (
        test_ratings
        .groupBy("user_id")
        .agg(count("*").alias("test_count"))
        .filter(col("test_count") >= precision_k)
        .select("user_id")
    )
    
    test_ratings = test_ratings.join(users_with_enough_test_items, on="user_id", how="inner")
    
    recommendations = recommendations.join(users_with_enough_test_items, on="user_id", how="inner")
    
    if "distance" in recommendations.columns:
        ranking_col = col("distance").asc()
    else:
        ranking_col = col("score").desc()
    
    ranked_recs = (
        recommendations
        .withColumn("rank", row_number().over(
            Window.partitionBy("user_id").orderBy(ranking_col)
        ))
        .filter(col("rank") <= max_k)
        .orderBy("user_id", "rank")
    )
    
    predictions_per_user = (
        ranked_recs
        .groupBy("user_id")
        .agg(collect_list("item_id").alias("predictions"))
    )
    
    relevant_per_user = (
        test_ratings
        .filter(col("rating") >= relevance_threshold)
        .groupBy("user_id")
        .agg(collect_list("item_id").alias("relevant"))
    )
    
    prediction_and_labels = (
        predictions_per_user
        .join(relevant_per_user, on="user_id", how="inner")
        .select("predictions", "relevant")
        .rdd
        .map(lambda row: (row.predictions, row.relevant))
    )
    
    metrics = RankingMetrics(prediction_and_labels)
    
    precision_at_k = metrics.precisionAt(precision_k)
    recall_at_k = metrics.recallAt(recall_k)
    ndcg_at_k = metrics.ndcgAt(ndcg_k)
    
    spark = recommendations.sparkSession
    return spark.createDataFrame(
        [(precision_at_k, recall_at_k, ndcg_at_k)],
        [f"avg_precision@{precision_k}", f"avg_recall@{recall_k}", f"avg_ndcg@{ndcg_k}"]
    )