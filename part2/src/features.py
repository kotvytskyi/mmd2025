"""Feature engineering for movies and users."""

import re
from pyspark.sql.functions import (
    min, max, col, explode, desc, when, array_contains, broadcast, array
)
from pyspark.ml.feature import CountVectorizer, VectorAssembler, Normalizer
from pyspark.ml.stat import Summarizer

from . import config


def safe_col_name(name: str) -> str:
    """Convert string to safe column name."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def build_movie_features(movies, embeddings_df, movies_enriched):
    """Build comprehensive movie feature vectors.
    
    Args:
        movies: DataFrame with movie metadata
        embeddings_df: DataFrame with overview embeddings
        movies_enriched: DataFrame with actor information
        
    Returns:
        DataFrame with movie_id and features_norm vector
    """
    # Normalize year
    year_stats = movies.agg(
        min("year").alias("min_year"),
        max("year").alias("max_year")
    ).collect()[0]
    
    min_year, max_year = year_stats["min_year"], year_stats["max_year"]
    
    movies = movies.withColumn(
        "year_norm",
        (col("year") - min_year) / (max_year - min_year)
    )
    
    # Add embeddings
    movies = movies.join(embeddings_df, on="item_id", how="left")
    
    # Extract top actors and create binary features
    top_actors = (
        movies_enriched
        .select(explode("actors").alias("actor"))
        .groupBy("actor")
        .count()
        .orderBy(desc("count"))
        .limit(config.TOP_N_ACTORS)
    )
    
    top_actors_list = [r["actor"] for r in top_actors.collect()]
    actor_cols = []
    
    for actor in top_actors_list:
        col_name = f"actor_{safe_col_name(actor)}"
        movies_enriched = movies_enriched.withColumn(
            col_name,
            when(array_contains(col("actors"), actor), 1.0).otherwise(0.0)
        )
        actor_cols.append(col_name)
    
    actors_features = movies_enriched.select("item_id", *actor_cols)
    movies = movies.join(actors_features, on="item_id", how="left")
    movies = movies.fillna(0.0, subset=actor_cols)
    
    # Vectorize genres
    cv = CountVectorizer(
        inputCol="genres",
        outputCol="tf",
        vocabSize=config.GENRE_VOCAB_SIZE,
        minDF=1,
        binary=True
    )
    
    cv_model = cv.fit(movies)
    movies = cv_model.transform(movies)
    
    # Assemble all features
    assembler = VectorAssembler(
        inputCols=["tf", "year_norm", "overview_emb"] + actor_cols,
        outputCol="features_raw",
        handleInvalid='skip'
    )
    
    movies = assembler.transform(movies)
    
    # Normalize features
    normalizer = Normalizer(
        inputCol="features_raw",
        outputCol="features_norm",
        p=2
    )
    
    movies_profiles = normalizer.transform(movies)
    
    return movies_profiles


def build_user_features(train_ratings, movies_profiles):
    """Build user feature vectors based on rated movies.
    
    Args:
        train_ratings: DataFrame with user ratings
        movies_profiles: DataFrame with movie features
        
    Returns:
        DataFrame with user_id and user_features_norm vector
    """
    movie_vecs = movies_profiles.select(
        col("item_id"),
        col("features_norm")
    )
    
    # Join ratings with movie features (broadcast movies for efficiency)
    user_movie_vectors = (
        train_ratings
        .join(broadcast(movie_vecs), on='item_id')
        .select("user_id", "rating", "features_norm")
    )
    
    # Aggregate using weighted mean
    user_profiles = (
        user_movie_vectors
        .groupBy("user_id")
        .agg(
            Summarizer.mean(
                col("features_norm"),
                weightCol=col("rating")
            ).alias("user_features")
        )
    )
    
    # Normalize user features
    normalizer = Normalizer(
        inputCol="user_features",
        outputCol="user_features_norm",
        p=2
    )
    
    user_profiles = normalizer.transform(user_profiles)
    
    return user_profiles
