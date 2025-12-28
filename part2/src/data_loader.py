"""Data loading and schema definitions."""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import substring, length, split, when, col, array, explode, desc

from . import config


def get_schemas():
    """Get PySpark schemas for data files."""
    schema_ratings = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("item_id", IntegerType(), False),
        StructField("rating", IntegerType(), False),
        StructField("timestamp", IntegerType(), False)
    ])

    schema_movies = StructType([
        StructField("item_id", IntegerType(), False),
        StructField("title", StringType(), False),
        StructField("genres", StringType(), False)
    ])

    schema_users = StructType([
        StructField("user_id", IntegerType(), False),
        StructField("gender", StringType(), False),
        StructField("age", StringType(), False),
        StructField("occupation", IntegerType(), False),
        StructField("zip_code", StringType(), False)
    ])
    
    return schema_ratings, schema_movies, schema_users


def load_ratings(spark: SparkSession):
    """Load train and test ratings."""
    schema_ratings, _, _ = get_schemas()
    
    train_ratings = spark.read.option("delimiter", "::").csv(
        config.RATINGS_TRAIN, schema=schema_ratings
    )
    test_ratings = spark.read.option("delimiter", "::").csv(
        config.RATINGS_TEST, schema=schema_ratings
    )
    
    return train_ratings, test_ratings


def load_movies(spark: SparkSession):
    """Load and preprocess movies data."""
    _, schema_movies, _ = get_schemas()
    
    movies = spark.read.option("delimiter", "::").csv(
        config.MOVIES, schema=schema_movies
    )
    
    movies = (
        movies
        .withColumn("year", substring("title", -5, 4).cast("int"))
        .withColumn("title", substring("title", 0, length("title") - 6))
        .withColumn("genres", split("genres", r"\|"))
    )
    
    return movies


def load_users(spark: SparkSession):
    """Load and preprocess users data."""
    _, _, schema_users = get_schemas()
    
    users = spark.read.option("delimiter", "::").csv(
        config.USERS, schema=schema_users
    )
    
    users = users.withColumn(
        "gender", when(col("gender") == "F", 0).otherwise(1)
    )
    
    return users


def load_embeddings(spark: SparkSession):
    """Load movie overview embeddings."""
    from pyspark.ml.functions import array_to_vector
    
    embeddings_df = (
        spark.read
        .option("header", True)
        .csv(config.EMBEDDINGS)
    )
    
    emb_cols = [f"emb_{i}" for i in range(config.EMBEDDING_DIM)]
    
    embeddings_df = (
        embeddings_df
        .select(
            col("movieId").cast("int").alias("item_id"),
            *[col(c).cast("float") for c in emb_cols]
        )
        .withColumn("overview_emb_array", array(*emb_cols))
    )
    
    embeddings_df = embeddings_df.withColumn(
        "overview_emb",
        array_to_vector(col("overview_emb_array"))
    ).select("item_id", "overview_emb")
    
    return embeddings_df


def load_enriched_movies(spark: SparkSession):
    """Load enriched movies data with actors."""
    movies_enriched = (
        spark.read
        .option("header", True)
        .option("escape", '"')
        .csv(config.MOVIES_ENRICHED)
        .select(
            col("movieId").alias("item_id"),
            split(col("actors"), r"\|").alias("actors")
        )
    )
    
    return movies_enriched
