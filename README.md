# Scalable Movie Recommendation Engine
### Distributed Collaborative Filtering with Apache Spark (ALS)

## 📌 Project Overview
This project implements a high-performance recommendation system using **Apache Spark's Alternating Least Squares (ALS)** algorithm. Designed for big data environments, the engine processes the **MovieLens 20M dataset** (over 20 million ratings) to deliver personalized "Top-20" movie recommendations for any given user.

The project demonstrates the ability to build, tune, and deploy a collaborative filtering model within a distributed computing environment (**Databricks**), ensuring the system remains scalable as the dataset grows.

---

## 🛠️ Technical Stack & Architecture
* **Engine:** PySpark (Spark MLlib)
* **Platform:** Databricks (Distributed Cluster Computing)
* **Algorithm:** Alternating Least Squares (ALS) — Matrix Factorization
* **Dataset:** MovieLens 20M (27,278 movies, 138,493 users, 20,000,263 ratings)

### Engineering Highlights:
* **Distributed Processing:** Utilized Spark DataFrames and RDD transformations to manage 20M+ records with high memory efficiency.
* **Hyperparameter Optimization:** Conducted systematic tuning to identify the optimal configuration:
    * **Rank:** 10 (Latent Factors)
    * **Max Iterations:** 15
    * **RegParam:** 0.01 (Regularization to prevent overfitting)
* **Cold Start Handling:** Implemented a robust "drop" strategy for the `coldStartStrategy` parameter to ensure model stability when encountering users or items with no prior history.

---

## 🚀 Key Performance Results
The model achieved high predictive accuracy by balancing latent factor complexity with regularization.

* **Primary Metric:** Root Mean Square Error (**RMSE**) of **0.8022**.
* **Scalability:** Successfully processed and predicted ratings across the entire 20M record matrix using Spark’s parallel processing capabilities.
* **User Output:** Generates real-time, personalized top-20 movie lists based on historical rating patterns and user-user similarity matrices.

---

## 📈 Key Insights
1.  **Matrix Factorization Power:** Demonstrated how ALS effectively decomposes the user-item interaction matrix into lower-dimensional latent features to uncover hidden preferences.
2.  **Infrastructure as Code:** Leveraged Databricks for managed Spark clusters, showcasing skills in cloud-based big data workflows.
3.  **Tuning for Accuracy:** Proved that subtle adjustments in `RegParam` were critical in lowering RMSE, highlighting the importance of regularization in high-dimensional datasets.

---

## 📂 Repository Structure
* `Movie_Recommendation_Engine.ipynb`: Full PySpark implementation including Data Loading, EDA, and Model Training.
* `configs/`: Model hyperparameter logs.
* `results/`: Sample top-20 recommendation outputs for selected user segments.
