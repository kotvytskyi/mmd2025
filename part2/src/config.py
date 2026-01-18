"""Configuration and constants."""

DATA_DIR = "./data"
RATINGS_TRAIN = f"{DATA_DIR}/ratings_train.dat"
RATINGS_TEST = f"{DATA_DIR}/ratings_test.dat"
MOVIES = f"{DATA_DIR}/movies.dat"
USERS = f"{DATA_DIR}/users.dat"
MOVIES_ENRICHED = f"{DATA_DIR}/movies_enriched.csv"
EMBEDDINGS = f"{DATA_DIR}/movies_overview_embeddings.csv"

SPARK_TMP_DIR = "/Users/oleksandr/spark-tmp"
DRIVER_MEMORY = "16g"
EXECUTOR_MEMORY = "16g"

EMBEDDING_DIM = 64
TOP_N_ACTORS = 30
GENRE_VOCAB_SIZE = 18

LSH_BUCKET_LENGTH = 2.0
LSH_NUM_HASH_TABLES = 5
LSH_THRESHOLD = 1.5

ALS_MAX_ITER = 20
ALS_REG_PARAM = 0.05
ALS_RANK = 15

TOP_K_PRECISION = 10
TOP_K_RECALL = 1000
TOP_K_NDCG = TOP_K_PRECISION
RELEVANCE_THRESHOLD = 4