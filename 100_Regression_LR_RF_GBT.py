# %% [markdown]
# # 회귀 분석 (LR → RF → GBT)
# - Linear Regression 베이스라인
# - RandomForest / GBT Regressor로 개선
# - 평가 지표: RMSE, MAE, R²
# - 산출물: baseline vs 개선 모델 비교표

# %%
import os
import sys
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.sql import SparkSession

IN_COLAB = "google.colab" in sys.modules
BASE = "/content" if IN_COLAB else os.getcwd()
SEED = 42

spark = SparkSession.builder.appName("Regression_LR_RF_GBT").getOrCreate()

# %% [markdown]
# ## 1. 데이터 로드: California Housing
#
# - 캘리포니아 주택 가격 데이터셋 (sklearn 제공)
# - 8개 feature, PRICE(target)

# %%
from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
pdf = pd.DataFrame(housing.data, columns=housing.feature_names)
pdf["PRICE"] = housing.target
spark_df = spark.createDataFrame(pdf)
spark_df.limit(5).toPandas()

# %%
# 스키마 확인
spark_df.printSchema()

# %% [markdown]
# ## 2. 전처리 파이프라인
#
# - VectorAssembler: 모든 feature를 하나의 벡터로 결합
# - StandardScaler: 평균 0, 분산 1로 정규화

# %%
# 수치형 feature 선택 (PRICE 제외)
feature_cols = [c for c in spark_df.columns if c != "PRICE"]

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# %%
# ===== 전처리 파이프라인 구성 =====
# VectorAssembler: 개별 피처 컬럼들을 하나의 벡터 컬럼("features")으로 결합
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# StandardScaler: 피처 정규화 (평균 0, 표준편차 1로 스케일링)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Pipeline: assembler → scaler 순서로 전처리 단계를 하나의 파이프라인으로 묶음
prep = Pipeline(stages=[assembler, scaler])

# ===== 전처리 파이프라인 실행 =====
# fit(): 학습 데이터로 스케일러 통계량(평균, 표준편차) 계산 → transform(): 변환 적용
df_ready = prep.fit(spark_df).transform(spark_df)

# 모델 학습에 필요한 컬럼만 선택하고, 컬럼명을 "features"로 변경
data = df_ready.select("scaled_features", "PRICE").withColumnRenamed("scaled_features", "features")
data.limit(5).toPandas()

# %% [markdown]
# ## 3. Train/Test 분리

# %%
train_data, test_data = data.randomSplit([0.8, 0.2], seed=SEED)

print(f"Train size: {train_data.count()}")
print(f"Test size: {test_data.count()}")

# %% [markdown]
# ## 4. 베이스라인: Linear Regression
#
# ### 회귀 평가 지표
# - **RMSE** (Root Mean Squared Error): 예측값과 실제값의 평균 제곱근 오차 (낮을수록 좋음)
# - **MAE** (Mean Absolute Error): 예측값과 실제값의 평균 절대 오차 (낮을수록 좋음)
# - **R²** (R-squared): 결정 계수, 모델의 설명력 (1에 가까울수록 좋음, 0~1)

# %%
# Linear Regression 모델 학습
lr = LinearRegression(featuresCol="features", labelCol="PRICE")
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

# %%
# 예측 결과 확인
lr_preds.select("features", "PRICE", "prediction").limit(5).toPandas()

# %%
# ===== 회귀 모델 평가 지표 객체 생성 =====
# RMSE (Root Mean Squared Error): 예측 오차의 제곱 평균의 제곱근 → 큰 오차에 민감
rmse_eval = RegressionEvaluator(labelCol="PRICE", predictionCol="prediction", metricName="rmse")

# MAE (Mean Absolute Error): 예측 오차의 절대값 평균 → 직관적 해석 용이
mae_eval = RegressionEvaluator(labelCol="PRICE", predictionCol="prediction", metricName="mae")

# R² (결정 계수): 모델이 데이터 분산을 설명하는 비율 → 1에 가까울수록 우수
r2_eval = RegressionEvaluator(labelCol="PRICE", predictionCol="prediction", metricName="r2")

# ===== 선형 회귀 모델 예측 결과(lr_preds)에 대해 평가 수행 =====
lr_rmse = rmse_eval.evaluate(lr_preds)  # 선형 회귀 RMSE 계산
lr_mae = mae_eval.evaluate(lr_preds)    # 선형 회귀 MAE 계산
lr_r2 = r2_eval.evaluate(lr_preds)      # 선형 회귀 R² 계산

# %%
print("=" * 60)
print("Linear Regression (Baseline) - Test Metrics:")
print("=" * 60)
print(f"  RMSE: {lr_rmse:.4f}")
print(f"  MAE:  {lr_mae:.4f}")
print(f"  R²:   {lr_r2:.4f}")
print("=" * 60)

# %% [markdown]
# ## 5. 개선 모델 1: Random Forest Regressor

# %%
rf = RandomForestRegressor(featuresCol="features", labelCol="PRICE", seed=SEED)
rf_model = rf.fit(train_data)
rf_preds = rf_model.transform(test_data)

# %%
rf_rmse = rmse_eval.evaluate(rf_preds)
rf_mae = mae_eval.evaluate(rf_preds)
rf_r2 = r2_eval.evaluate(rf_preds)

# %%
print("=" * 60)
print("Random Forest Regressor - Test Metrics:")
print("=" * 60)
print(f"  RMSE: {rf_rmse:.4f}")
print(f"  MAE:  {rf_mae:.4f}")
print(f"  R²:   {rf_r2:.4f}")
print("=" * 60)

# %%
# Feature Importance
print("\nRandom Forest Feature Importances:")
for feature, importance in zip(feature_cols, rf_model.featureImportances):
    print(f"  {feature:20s}: {importance:.4f}")

# %% [markdown]
# ## 6. 개선 모델 2: Gradient Boosted Trees Regressor

# %%
gbt = GBTRegressor(featuresCol="features", labelCol="PRICE", seed=SEED)
gbt_model = gbt.fit(train_data)
gbt_preds = gbt_model.transform(test_data)

# %%
gbt_rmse = rmse_eval.evaluate(gbt_preds)
gbt_mae = mae_eval.evaluate(gbt_preds)
gbt_r2 = r2_eval.evaluate(gbt_preds)

# %%
print("=" * 60)
print("Gradient Boosted Trees Regressor - Test Metrics:")
print("=" * 60)
print(f"  RMSE: {gbt_rmse:.4f}")
print(f"  MAE:  {gbt_mae:.4f}")
print(f"  R²:   {gbt_r2:.4f}")
print("=" * 60)

# %%
# Feature Importance
print("\nGBT Feature Importances:")
for feature, importance in zip(feature_cols, gbt_model.featureImportances):
    print(f"  {feature:20s}: {importance:.4f}")

# %% [markdown]
# ## 7. 최종 비교

# %%
# 결과 요약
results = {
    "Model": ["Linear Regression", "Random Forest", "GBT"],
    "RMSE": [lr_rmse, rf_rmse, gbt_rmse],
    "MAE": [lr_mae, rf_mae, gbt_mae],
    "R²": [lr_r2, rf_r2, gbt_r2]
}

results_df = pd.DataFrame(results)
results_df

# %%
spark.stop()
