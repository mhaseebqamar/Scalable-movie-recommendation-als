# Scalable Movie Recommendation Engine

📌 Project Overview

This project implements a collaborative filtering recommendation system using Apache Spark's ALS (Alternating Least Squares) algorithm. It processes a dataset of 20 million movie ratings to predict user preferences and generate personalized top-20 movie recommendations.

🛠️ Technical Stack

Engine: PySpark (Spark MLlib)

Platform: Databricks

Algorithm: Alternating Least Squares (ALS)

Dataset: MovieLens 20M (27,278 movies, 20,000,263 ratings)

🚀 Key Features

Big Data Processing: Efficiently handled 20M+ records using Spark DataFrames and distributed computing.

Hyperparameter Tuning: Optimized the model across multiple parameters (Rank: 10, Max Iter iterations: 15, RegParam: 0.01).

Model Performance: Achieved a Root Mean Square Error (RMSE) of 0.8022, ensuring high-quality recommendation accuracy.

Cold Start Strategy: Implemented the "drop" strategy to handle new users/items during the prediction phase.

📈 Results

The system successfully generates 20 tailored movie recommendations for any given user ID based on their historical rating patterns and similarity to other users in the dataset.
