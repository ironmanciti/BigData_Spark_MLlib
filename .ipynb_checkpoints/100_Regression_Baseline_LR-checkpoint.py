# %% [markdown]
# Day2-3교시: 회귀 베이스라인 (Linear Regression)
# - RMSE, MAE, R2, seed 고정
# - 산출물: 회귀 결과표

# %%
import os
import sys
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession

IN_COLAB = "google.colab" in sys.modules
BASE = "/content" if IN_COLAB else os.getcwd()
SEED = 42

spark = SparkSession.builder.appName("Day2_Regression_Baseline").getOrCreate()

# %%
try:
    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    housing = fetch_california_housing()
    pdf = pd.DataFrame(housing.data, columns=housing.feature_names)
    pdf["PRICE"] = housing.target
    spark_df = spark.createDataFrame(pdf)
except Exception:
    csv_path = os.path.join(BASE, "TestData", "retail_sales_dataset.csv")
    spark_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(csv_path)
    if "PRICE" not in spark_df.columns and "target" in spark_df.columns:
        spark_df = spark_df.withColumnRenamed("target", "PRICE")

# %%
feature_cols = [c for c in spark_df.columns if c != "PRICE" and spark_df.schema[c].dataType.simpleString() in ("double", "int")]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
from pyspark.ml import Pipeline
prep = Pipeline(stages=[assembler, scaler])
df_ready = prep.fit(spark_df).transform(spark_df)
data = df_ready.select("scaled_features", "PRICE").withColumnRenamed("scaled_features", "features")

train_data, test_data = data.randomSplit([0.8, 0.2], seed=SEED)

# %%
lr = LinearRegression(featuresCol="features", labelCol="PRICE")
lr_model = lr.fit(train_data)
preds = lr_model.transform(test_data)

# %%
rmse_eval = RegressionEvaluator(labelCol="PRICE", predictionCol="prediction", metricName="rmse")
mae_eval = RegressionEvaluator(labelCol="PRICE", predictionCol="prediction", metricName="mae")
r2_eval = RegressionEvaluator(labelCol="PRICE", predictionCol="prediction", metricName="r2")

rmse = rmse_eval.evaluate(preds)
mae = mae_eval.evaluate(preds)
r2 = r2_eval.evaluate(preds)

print("Linear Regression - Test: RMSE:", rmse, " MAE:", mae, " R2:", r2)

# %%
print("=== 회귀 결과 (산출물에 기록) ===")
print("model,rmse,mae,r2")
print(f"LinearRegression,{rmse:.4f},{mae:.4f},{r2:.4f}")

spark.stop()
