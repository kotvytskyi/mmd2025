"""Configuration and constants."""

# Data paths
DATA_DIR = "./data"
RATINGS_TRAIN = f"{DATA_DIR}/ratings_train.dat"
RATINGS_TEST = f"{DATA_DIR}/ratings_test.dat"
MOVIES = f"{DATA_DIR}/movies.dat"
USERS = f"{DATA_DIR}/users.dat"
MOVIES_ENRICHED = f"{DATA_DIR}/movies_enriched.csv"
EMBEDDINGS = f"{DATA_DIR}/movies_overview_embeddings.csv"

# Spark config
SPARK_TMP_DIR = "/Users/oleksandr/spark-tmp"
DRIVER_MEMORY = "16g"
EXECUTOR_MEMORY = "16g"

# Feature engineering
EMBEDDING_DIM = 64
TOP_N_ACTORS = 30
GENRE_VOCAB_SIZE = 18

# LSH parameters
LSH_BUCKET_LENGTH = 2.0
LSH_NUM_HASH_TABLES = 5
LSH_THRESHOLD = 1.5

# ALS parameters
ALS_MAX_ITER = 20
ALS_REG_PARAM = 0.05
ALS_RANK = 15

# Recommendation parameters
TOP_K = 1000
RELEVANCE_THRESHOLD = 4

# Hybrid fusion
HYBRID_ALPHA = 0.5  # Weight for ALS (1-alpha for content-based)
