# Movie Recommendation with Spark ALS on MovieLens 20M

An individual MSc project (SGH Warsaw School of Economics, MSc Advanced Analytics & Big Data, 2025/26) that builds a collaborative-filtering movie recommender with Apache Spark's Alternating Least Squares (ALS) on Databricks, using the MovieLens 20M dataset.

## Project context

The goal was to work with a dataset large enough to need Spark, tune a matrix factorisation model systematically, and then use it to produce personalised recommendations for a new user. Everything runs on Databricks using the built-in copy of MovieLens.

## What this project does

* Loads MovieLens 20M from the Databricks datasets store with explicit Spark schemas: **20,000,263 ratings** and **27,278 movies** (ratings 0.5–5.0, mean 3.53, standard deviation 1.05)
* Splits the ratings randomly 80/20 into 15,999,973 training and 4,000,290 test ratings
* Trains Spark MLlib ALS models over a **27-configuration grid**: rank (5, 10, 15) × max iterations (5, 10, 15) × regularisation (0.01, 0.1, 0.5), with `coldStartStrategy="drop"`
* Evaluates every configuration by RMSE on the test ratings and visualises the effect of each hyperparameter (line plots and RMSE heatmaps)
* Adds 14 of my own ratings as a new user, retrains the best configuration and scores all 27,264 movies I had not rated to produce a top-20 list

## Key findings

**Best configuration: rank 10, 15 iterations, regularisation 0.01, giving RMSE 0.802**

| Rank | Max iter | Reg | RMSE |
|---|---|---|---|
| 10 | 15 | 0.01 | 0.802 |
| 10 | 10 | 0.01 | 0.804 |
| 15 | 15 | 0.10 | 0.805 |
| 10 | 15 | 0.10 | 0.807 |
| 15 | 10 | 0.10 | 0.808 |

* **For reference**, the standard deviation of all ratings is 1.05, which is roughly the error of always predicting the average rating. ALS reduces that to about 0.80.
* **Regularisation mattered most.** Every model with regularisation 0.5 was much worse (RMSE about 1.00), whatever the rank or number of iterations. Between 0.01 and 0.1 the differences were small (0.80–0.82).
* **More iterations helped slightly**, and a larger rank did not consistently help: rank 15 was not better than rank 10.

**The personalised recommendations exposed a real weakness**

* The top-20 list for my test user was dominated by obscure titles, and the predicted ratings went as high as **11.27 on a 5-point scale**.
* This happens because the lowest regularisation (0.01), chosen because it gave the best RMSE, lets movies with very few ratings get extreme latent factors, and the predictions were not capped to the rating scale. The configuration that is best on average RMSE is not automatically good for ranking recommendations for one user.

## Pipeline overview

```
MovieLens 20M (Databricks datasets) → Spark DataFrames with explicit schemas
    → 80/20 random split
    → 27-configuration ALS grid search (RMSE on test ratings)
    → best model → add new user's ratings → retrain → score unrated movies → top 20
```

## Tech stack

* Databricks, Apache Spark (PySpark)
* Spark MLlib: `ALS`, `RegressionEvaluator`
* pandas: results table
* Matplotlib, seaborn: hyperparameter plots and heatmaps

## Project structure

```
Scalable-movie-recommendation-als/
├── ALS_Movie_Recommendation.ipynb   # Full notebook: loading → grid search → best model → personalised top 20
└── README.md
```

The notebook runs on Databricks and reads the data from `/databricks-datasets/cs110x/ml-20m/`, so no data files are stored in this repository.

## Honest limitations

* **No separate validation set.** The best configuration was chosen using the same test ratings it is reported on, so the 0.802 RMSE is slightly optimistic.
* **Random rather than time-based split.** Some test ratings come from earlier than training ratings, which is easier than predicting future behaviour.
* **RMSE only.** No ranking metrics (precision@k, recall@k, NDCG), which matter more for a recommender than rating error.
* **No filtering of rarely rated movies and no clipping of predictions**, which led to the unrealistic top-20 described above.
* **No popularity or bias baseline** beyond the rating standard deviation.
