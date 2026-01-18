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
    schema_ratings = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("movie_id", IntegerType(), False),
        StructField("rating", IntegerType(), False),
        StructField("timestamp", IntegerType(), False)
    ])
    
    ratings = spark.read.option("delimiter", "::").csv(input_path, schema=schema_ratings)
    
    w = Window.partitionBy("user_id").orderBy(col("timestamp").desc())
    ranked = ratings.withColumn("rn", row_number().over(w))
    
    counts = ratings.groupBy("user_id").count()
    
    joined = ranked.join(counts, "user_id")
    
    train = joined.filter(col("rn") > col("count") * test_ratio).drop("rn", "count")
    test = joined.filter(col("rn") <= col("count") * test_ratio).drop("rn", "count")
    
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
