"""Example usage of the recommendation system modules."""

from pyspark.sql import SparkSession

# Import our modules
from src import config
from src.data_loader import load_ratings, load_movies, load_embeddings, load_enriched_movies
from src.features import build_movie_features, build_user_features
from src.lsh_recommender import LSHRecommender
from src.als_recommender import ALSRecommender
from src.hybrid_recommender import HybridRecommender
from src.evaluation import evaluate_recommendations, coverage_stats


def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("MMDS") \
        .master("local[*]") \
        .config("spark.local.dir", config.SPARK_TMP_DIR) \
        .config("spark.driver.memory", config.DRIVER_MEMORY) \
        .config("spark.executor.memory", config.EXECUTOR_MEMORY) \
        .getOrCreate()
    
    # Load data
    print("Loading data...")
    train_ratings, test_ratings = load_ratings(spark)
    movies = load_movies(spark)
    embeddings_df = load_embeddings(spark)
    movies_enriched = load_enriched_movies(spark)
    
    # Build features
    print("Building features...")
    movies_profiles = build_movie_features(movies, embeddings_df, movies_enriched)
    user_profiles = build_user_features(train_ratings, movies_profiles)
    
    # --- LSH Recommender ---
    print("\n=== LSH Recommender ===")
    lsh_rec = LSHRecommender()
    lsh_rec.fit(movies_profiles)
    lsh_recs = lsh_rec.recommend(user_profiles, movies_profiles, train_ratings)
    
    lsh_metrics = evaluate_recommendations(lsh_recs, test_ratings)
    print("LSH Metrics:")
    lsh_metrics.show()
    
    # --- ALS Recommender ---
    print("\n=== ALS Recommender ===")
    als_rec = ALSRecommender()
    als_rec.fit(train_ratings)
    
    test_users = test_ratings.select("user_id").distinct()
    als_recs = als_rec.recommend(test_users, train_ratings)
    
    als_metrics = evaluate_recommendations(als_recs, test_ratings)
    print("ALS Metrics:")
    als_metrics.show()
    
    # --- Hybrid Recommender ---
    print("\n=== Hybrid Recommender ===")
    hybrid_rec = HybridRecommender(alpha=0.5)
    hybrid_recs = hybrid_rec.fuse(lsh_recs, als_recs)
    
    hybrid_metrics = evaluate_recommendations(hybrid_recs, test_ratings)
    print("Hybrid Metrics:")
    hybrid_metrics.show()
    
    # Coverage stats
    print("\n=== Coverage Statistics ===")
    stats = coverage_stats(hybrid_recs, test_ratings)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    spark.stop()


if __name__ == "__main__":
    main()
