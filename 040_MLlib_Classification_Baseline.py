# %% [markdown]
# #  MLlib 파이프라인 & 분류 베이스라인
# - Estimator vs Transformer, Pipeline, Fit/Transform
# - StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
# - Logistic Regression 베이스라인 구축
# - 평가 지표: AUC, PR, Confusion Matrix
# - 산출물: 공통 파이프라인 템플릿 코드, baseline 결과표

# %%
import os
import sys
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql import SparkSession

IN_COLAB = "google.colab" in sys.modules
BASE = "/content" if IN_COLAB else os.getcwd()
CSV_PATH = os.path.join(BASE, "Social_Network_Ads.csv")
SEED = 42

spark = SparkSession.builder.appName("MLlib_Classification_Baseline").getOrCreate()

# %% [markdown]
# ## 1. 데이터 로드 및 확인

# %%
# CSV 파일을 DataFrame으로 읽기
df = (spark.read.format("csv")
    .option("header", "true")             # 첫 번째 행을 헤더(컬럼명)로 사용
    .option("inferSchema", "true")    # 데이터 타입 자동 추론 (문자열, 숫자 등)
    .load(CSV_PATH))                     # 파일 로드

# 상위 3개 행 출력 (데이터 확인용)
df.limit(3).toPandas()

# %%
# 데이터 스키마 확인
df.printSchema()

# %% [markdown]
# ## 2. MLlib 파이프라인 구조
#
# ### Transformer vs Estimator
# - **Transformer**: transform() 메서드로 DataFrame을 변환 (예: VectorAssembler)
# - **Estimator**: fit() 메서드로 학습 후 Transformer 생성 (예: StringIndexer, StandardScaler)
#
# ### 단계별 구성
# 1. **StringIndexer**: 범주형 컬럼(Gender) → 숫자 인덱스 (fit으로 vocabulary 결정)
# 2. **OneHotEncoder**: 인덱스 → 희소 벡터 (원-핫 인코딩)
# 3. **VectorAssembler**: 여러 컬럼 → 단일 feature 벡터 (Transformer)
# 4. **StandardScaler**: 평균 0, 분산 1로 정규화 (fit으로 mean/std 결정)

# %%
# 1. StringIndexer: 문자열을 숫자로 변환 (Gender → Gender_idx)
indexer = StringIndexer(inputCol="Gender", outputCol="Gender_idx") \
    .setHandleInvalid("keep")  # 학습 시 없던 값이 나와도 유지 (에러 방지)
# 예: Male → 0, Female → 1

# 2. OneHotEncoder: 범주형 숫자를 원-핫 벡터로 변환 (Gender_idx → Gender_ohe)
encoder = OneHotEncoder(inputCols=["Gender_idx"], outputCols=["Gender_ohe"])
# 예: 0 → [1, 0], 1 → [0, 1]

# 3. VectorAssembler: 여러 피처를 하나의 벡터로 결합 (features 컬럼 생성)
assembler = VectorAssembler(
    inputCols=["Age", "EstimatedSalary", "Gender_ohe"],  # 입력: 나이, 연봉, 성별(원-핫)
    outputCol="features"                                  # 출력: 하나의 벡터
)
# 예: [19, 19000, [1,0]] → [19.0, 19000.0, 1.0, 0.0]

# 4. StandardScaler: 피처를 표준화 (평균 0, 분산 1로 스케일링)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# %% [markdown]
# ### Pipeline으로 순서 정의
# - 모든 단계를 Pipeline에 넣으면 fit() 한 번에 전 단계 학습
# - transform()으로 새 데이터에도 동일한 변환 적용

# %%
# Pipeline 생성: 전처리 단계들을 하나로 묶음
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])
# 순서: StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler

# Pipeline 학습: 각 단계를 순차적으로 학습
model = pipeline.fit(df)
# 내부 동작:
# 1. indexer.fit(df) → Gender 고유값 파악 (Male, Female)
# 2. encoder는 학습 불필요 (Transformer)
# 3. assembler는 학습 불필요 (Transformer)
# 4. scaler.fit() → features의 평균, 표준편차 계산

# 학습된 Pipeline으로 데이터 변환
transformed = model.transform(df)
# 내부 동작:
# 1. indexer_model.transform(df) → Gender를 숫자로 변환
# 2. encoder.transform() → 숫자를 원-핫 벡터로 변환
# 3. assembler.transform() → 모든 피처를 하나의 벡터로 결합
# 4. scaler_model.transform() → 학습된 평균/표준편차로 표준화

# %%
# 변환 결과 확인 (원본 데이터와 최종 변환된 피처 비교)
transformed.select("Age", "EstimatedSalary", "Gender", "scaled_features", "Purchased") \
    .limit(5).toPandas()

# %% [markdown]
# ## 3. 분류 베이스라인 구축 (Logistic Regression)
#
# ### Train/Test 분리

# %%
# 파이프라인 적용 후 ML 모델 학습용 데이터 준비

# 파이프라인으로 전처리 수행 (학습 + 변환)
df_ready = pipeline.fit(df).transform(df)

# ML 모델에 필요한 컬럼만 선택 및 이름 변경
data = df_ready.select("scaled_features", "Purchased") \
    .withColumnRenamed("scaled_features", "features")
# - scaled_features: 표준화된 피처 벡터
# - Purchased: 타겟 변수 (예측할 값, 0 또는 1)
# - withColumnRenamed: 컬럼명을 "scaled_features" → "features"로 변경
#   (PySpark ML 모델은 기본적으로 "features" 컬럼명을 기대함)
data.limit(5).toPandas()

# %%
# Train/Test split (80:20, seed 고정)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=SEED)

print(f"Train size: {train_data.count()}")
print(f"Test size: {test_data.count()}")

# %% [markdown]
# ### Logistic Regression 모델 학습

# %%
lr = LogisticRegression(featuresCol="features", labelCol="Purchased")
lr_model = lr.fit(train_data)

# %%
# Test 데이터에 예측
predictions = lr_model.transform(test_data)
# 예측 결과 확인
predictions.limit(5).toPandas()

# %% [markdown]
# ### 평가 지표 계산
#
# - **AUC (ROC)**: ROC 곡선 아래 면적 (0~1, 높을수록 좋음)
# - **AUC (PR)**: Precision-Recall 곡선 아래 면적
# - **Accuracy**: 정확도 (맞춘 비율)

# %%
# 이진 분류 평가 지표 설정 및 계산

# 1. AUC-ROC 평가기 생성 (ROC 곡선 아래 면적)
auc_eval = BinaryClassificationEvaluator(
    labelCol="Purchased",                     # 실제 라벨 (정답)
    rawPredictionCol="rawPrediction",  # 모델의 원시 예측값 (확률 변환 전)
    metricName="areaUnderROC"          # 평가 지표: AUC-ROC
)

# 2. AUC-PR 평가기 생성 (Precision-Recall 곡선 아래 면적)
pr_eval = BinaryClassificationEvaluator(
    labelCol="Purchased",                     # 실제 라벨
    rawPredictionCol="rawPrediction",  # 모델의 원시 예측값
    metricName="areaUnderPR"           # 평가 지표: AUC-PR
)

# 3. 정확도 평가기 생성
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="Purchased",              # 실제 라벨
    predictionCol="prediction",        # 모델의 최종 예측값 (0 또는 1)
    metricName="accuracy"              # 평가 지표: 정확도 (맞춘 비율)
)

# 평가 지표 계산
auc = auc_eval.evaluate(predictions)           # AUC-ROC 값 계산 (0~1, 높을수록 좋음)
pr_area = pr_eval.evaluate(predictions)        # AUC-PR 값 계산 (0~1, 높을수록 좋음)
accuracy = accuracy_eval.evaluate(predictions) # 정확도 계산 (0~1, 높을수록 좋음)

# %%
print("=" * 50)
print("기본 로지스틱 회귀 모델 - 테스트 성능 지표:")
print("=" * 50)
print(f"  AUC (ROC):  {auc:.4f}")
print(f"  AUC (PR):   {pr_area:.4f}")
print(f"  Accuracy:   {accuracy:.4f}")
print("=" * 50)

# %% [markdown]
# ### Confusion Matrix

# %%
# 실제 vs 예측 분포
predictions.groupBy("Purchased", "prediction").count().orderBy("Purchased", "prediction").toPandas()

# %% [markdown]
# ## 정리
#
# **학습한 내용:**
# 1. MLlib 파이프라인 구조 (Transformer/Estimator)
# 2. 전처리 단계 (StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler)
# 3. Logistic Regression 베이스라인 모델 구축
# 4. 평가 지표 (AUC, PR, Accuracy, Confusion Matrix)
#
# **다음 단계:**
# - 050_CrossValidator_Tuning.py에서 하이퍼파라미터 튜닝으로 성능 개선

# %%
spark.stop()
