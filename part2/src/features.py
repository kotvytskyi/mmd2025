import re
from pyspark.sql.functions import (
    min, max, col, explode, desc, when, array_contains, broadcast, array
)
from pyspark.ml.feature import CountVectorizer, VectorAssembler, Normalizer
from pyspark.ml.stat import Summarizer

from . import config


def safe_col_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def build_movie_features(movies, embeddings_df, movies_enriched):
    year_stats = movies.agg(
        min("year").alias("min_year"),
        max("year").alias("max_year")
    ).collect()[0]
    
    min_year, max_year = year_stats["min_year"], year_stats["max_year"]
    
    movies = movies.withColumn(
        "year_norm",
        (col("year") - min_year) / (max_year - min_year)
    )
    
    movies = movies.join(embeddings_df, on="item_id", how="left")
    
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
    
    cv = CountVectorizer(
        inputCol="genres",
        outputCol="tf",
        binary=True
    )
    
    cv_model = cv.fit(movies)
    movies = cv_model.transform(movies)
    
    assembler = VectorAssembler(
        inputCols=["tf", "year_norm", "overview_emb"] + actor_cols,
        outputCol="features_raw",
        handleInvalid='skip'
    )
    
    movies = assembler.transform(movies)
    
    normalizer = Normalizer(
        inputCol="features_raw",
        outputCol="features_norm",
        p=2
    )
    
    movies_profiles = normalizer.transform(movies)
    
    return movies_profiles


def build_user_features(train_ratings, movies_profiles):
    movie_vecs = movies_profiles.select(
        col("item_id"),
        col("features_norm")
    )
    
    user_movie_vectors = (
        train_ratings
        .filter(col("rating") >= config.RELEVANCE_THRESHOLD)
        .join(broadcast(movie_vecs), on='item_id') # this way we hint spark to avoid shuffling large train_ratings array
        .select("user_id", "rating", "features_norm")
    )

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
    
    normalizer = Normalizer(
        inputCol="user_features",
        outputCol="user_features_norm",
        p=2
    )
    
    user_profiles = normalizer.transform(user_profiles)
    
    return user_profiles
