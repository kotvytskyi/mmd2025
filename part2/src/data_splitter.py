"""
Data splitting utilities for train/test split of ratings data.
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import col, row_number, concat_ws
from pyspark.sql.window import Window


def split_ratings_by_user(
    spark: SparkSession,
    input_path: str,
    train_output_path: str,
    test_output_path: str,
    test_ratio: float = 0.2
) -> tuple:
    """
    Split ratings data into train and test sets based on temporal split per user.
    
    For each user, the most recent test_ratio of ratings go to test set,
    and the remaining ratings go to train set.
    
    Args:
        spark: SparkSession instance
        input_path: Path to the ratings.dat file
        train_output_path: Path to save training data
        test_output_path: Path to save test data
        test_ratio: Proportion of most recent ratings per user for test set (default: 0.2)
    
    Returns:
        tuple: (train_df, test_df) DataFrames
    """
    # Define schema for ratings data
    schema_ratings = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("movie_id", IntegerType(), False),
        StructField("rating", IntegerType(), False),
        StructField("timestamp", IntegerType(), False)
    ])
    
    # Read ratings data
    ratings = spark.read.option("delimiter", "::").csv(input_path, schema=schema_ratings)
    
    # Rank ratings by timestamp (most recent first) within each user
    w = Window.partitionBy("user_id").orderBy(col("timestamp").desc())
    ranked = ratings.withColumn("rn", row_number().over(w))
    
    # Count total ratings per user
    counts = ratings.groupBy("user_id").count()
    
    # Join ranked ratings with counts
    joined = ranked.join(counts, "user_id")
    
    # Split: test set contains the most recent test_ratio of ratings per user
    train = joined.filter(col("rn") > col("count") * test_ratio).drop("rn", "count")
    test = joined.filter(col("rn") <= col("count") * test_ratio).drop("rn", "count")
    
    # Save in the same format as input (::-delimited)
    train_out = train.select(
        concat_ws(
            "::",
            col("user_id"),
            col("movie_id"),
            col("rating"),
            col("timestamp")
        ).alias("value")
    )
    
    test_out = test.select(
        concat_ws(
            "::",
            col("user_id"),
            col("movie_id"),
            col("rating"),
            col("timestamp")
        ).alias("value")
    )
    
    # Write to disk
    train_out \
        .write \
        .mode("overwrite") \
        .option("header", "false") \
        .csv(train_output_path)
    
    test_out \
        .write \
        .mode("overwrite") \
        .option("header", "false") \
        .csv(test_output_path)
    
    return train, test


if __name__ == "__main__":
    # Example usage
    spark = SparkSession.builder \
        .appName("MMDS_DataSplit") \
        .master("local[*]") \
        .getOrCreate()
    
    train_df, test_df = split_ratings_by_user(
        spark=spark,
        input_path="../data/ratings.dat",
        train_output_path="../data/ratings_train.dat",
        test_output_path="../data/ratings_test.dat",
        test_ratio=0.2
    )
    
    print(f"Train set size: {train_df.count()}")
    print(f"Test set size: {test_df.count()}")
    
    spark.stop()
