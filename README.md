# MMDS Final Project 2025

## Analyze User-Movie Data Stream and Build a Hybrid Recommendation System

**Data:** MovieLens 1M dataset (includes interactions and content details)

**Environment:** Jupyter/Colab

---

## Progress Tracker

### Part 1: Streams (20/80 points)

- [ ] Read user-item data (`u.data` file) as a stream (line by line)
- [ ] Create a random sample (~10% of movies)
- [ ] Count approximate number of different users

**Technologies:** Python with only `random` library (no other imports allowed)

---

### Part 2: RecSys (60/80 points)

- [x] (i) Load, process and prepare the data in Spark
- [x] (i) Basic EDA
- [x] (ii) Build Collaborative Filtering Component (ALS, etc.)
- [x] (iii) Build Content-Based Component (TF-IDF, LSH, etc.)
- [x] (iv) Build recommendation fusion strategy
- [ ] (v) Evaluate system using Precision@K, Recall@K, NDCG@K
- [ ] (v) Evaluate training and inference time

**Technologies:** Apache Spark (PySpark), Spark MLlib (ALS, TF-IDF, hashing, LSH). No Pandas/Scikit-learn for main pipeline.

---

### Additional (+5 points)

- [ ] Implement hyperparameter tuning using MLlib pipelines
- [ ] Build second layer ranking / custom scoring model (e.g., GBT)

---

## EDA Summary

- **Users:** 6,040 users with demographics (gender, age, occupation, zip code)
- **Movies:** 3,883 movies with genres and release years (1919-2000)
- **Ratings:** 1,000,209 ratings (scale 1-5, mean: 3.58)
- **Key findings:**
  - Most active user: 2,314 ratings
  - Most rated movie: American Beauty (3,428 ratings)
  - Top genres by rating count: Comedy, Drama, Action
  - Top genres by avg rating: Film-Noir, Documentary, War