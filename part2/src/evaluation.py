"""Evaluation metrics for recommendation systems."""

from pyspark.sql.functions import col, sum, avg, count, lit, when, min, max

from . import config


def evaluate_recommendations(recommendations, test_ratings, top_k=config.TOP_K, relevance_threshold=config.RELEVANCE_THRESHOLD):
    """Evaluate recommendations using precision and recall.
    
    Args:
        recommendations: DataFrame with user_id, item_id, and optionally rank
        test_ratings: DataFrame with user_id, item_id, rating
        top_k: Number of recommendations per user (for precision calculation)
        relevance_threshold: Minimum rating to consider an item relevant
        
    Returns:
        DataFrame with avg_precision and avg_recall
    """
    # Join recommendations with test ratings
    recs_on_rated = (
        recommendations
        .join(test_ratings, on=["user_id", "item_id"], how="inner")
    )
    
    # Mark relevant items (rating >= threshold)
    eval_df = (
        recs_on_rated
        .withColumn("relevant", (col("rating") >= relevance_threshold).cast("int"))
    )
    
    # Calculate per-user metrics
    user_metrics = (
        eval_df
        .groupBy("user_id")
        .agg(
            (sum("relevant") / lit(top_k)).alias("precision"),
            sum("relevant").alias("hits"),
        )
        .join(
            test_ratings
            .filter(col("rating") >= relevance_threshold)
            .groupBy("user_id")
            .count()
            .withColumnRenamed("count", "total_relevant"),
            on="user_id",
            how="left"
        )
        .fillna(0)
        .withColumn(
            "recall",
            when(col("total_relevant") > 0,
                 col("hits") / col("total_relevant"))
            .otherwise(lit(0))
        )
    )
    
    # Calculate average metrics
    avg_metrics = user_metrics.agg(
        avg("precision").alias(f"avg_precision@{top_k}"),
        avg("recall").alias(f"avg_recall@{top_k}")
    )
    
    return avg_metrics


def coverage_stats(recommendations, test_ratings):
    """Calculate recommendation coverage statistics.
    
    Args:
        recommendations: DataFrame with user_id, item_id
        test_ratings: DataFrame with user_id, item_id
        
    Returns:
        Dictionary with coverage statistics
    """
    # Users with recommendations
    total_users_with_recs = recommendations.select("user_id").distinct().count()
    total_test_users = test_ratings.select("user_id").distinct().count()
    
    # Recommendations per user
    recs_per_user = (
        recommendations
        .groupBy("user_id")
        .count()
        .agg(
            avg("count").alias("avg_recs_per_user"),
            min("count").alias("min_recs"),
            max("count").alias("max_recs")
        )
        .collect()[0]
    )
    
    # Overlap with test set
    overlap = (
        recommendations
        .join(test_ratings, on=["user_id", "item_id"], how="inner")
        .groupBy("user_id")
        .count()
        .agg(avg("count").alias("avg_overlap"))
        .collect()[0]
    )
    
    return {
        "users_with_recs": total_users_with_recs,
        "total_users": total_test_users,
        "coverage_pct": 100.0 * total_users_with_recs / total_test_users if total_test_users > 0 else 0,
        "avg_recs_per_user": recs_per_user["avg_recs_per_user"],
        "min_recs": recs_per_user["min_recs"],
        "max_recs": recs_per_user["max_recs"],
        "avg_overlap_with_test": overlap["avg_overlap"]
    }
