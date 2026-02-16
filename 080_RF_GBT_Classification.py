# %% [markdown]
# # 트리 기반 분류 (RF / GBT)
# - RandomForestClassifier, GBTClassifier, feature importance
# - 산출물: 모델별 성능 비교표

# %%
import os
import sys
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql import SparkSession

IN_COLAB = "google.colab" in sys.modules
BASE = "/content" if IN_COLAB else os.getcwd()
CSV_PATH = os.path.join(BASE,  "Social_Network_Ads.csv")
SEED = 42

spark = SparkSession.builder.appName("RF_GBT").getOrCreate()

# %%
# 1. CSV 파일을 DataFrame으로 읽기
df = (spark.read.format("csv")
    .option("header", "true")          # 첫 번째 행을 헤더(컬럼명)로 사용
    .option("inferSchema", "true")  # 데이터 타입 자동 추론 (Age→int, Gender→string 등)
    .load(CSV_PATH))

# 2. 전처리 단계 정의
# StringIndexer: 문자열을 숫자로 변환 (Gender → Gender_idx)
indexer = StringIndexer(inputCol="Gender", outputCol="Gender_idx") \
    .setHandleInvalid("keep")  # 학습에 없던 값이 나와도 처리 (에러 방지)
# 예: Male → 0, Female → 1

# OneHotEncoder: 범주형 숫자를 원-핫 벡터로 변환 (Gender_idx → Gender_ohe)
encoder = OneHotEncoder(inputCols=["Gender_idx"], outputCols=["Gender_ohe"])
# 예: 0 → [1, 0], 1 → [0, 1]

# VectorAssembler: 여러 피처를 하나의 벡터로 결합
assembler = VectorAssembler(
    inputCols=["Age", "EstimatedSalary", "Gender_ohe"],  # 입력: 나이, 연봉, 성별(원-핫)
    outputCol="features"                                  # 출력: features 벡터
)
# 예: Age=19, Salary=19000, Gender=[1,0] → [19.0, 19000.0, 1.0, 0.0]

# StandardScaler: 피처 표준화 (평균 0, 분산 1)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# 3. 전처리 파이프라인 생성 및 적용 - 모든 전처리 단계를 순서대로 묶음
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])

# 파이프라인 학습(fit) 후 데이터 변환(transform)
# fit: StringIndexer와 StandardScaler가 데이터로부터 통계 학습
# transform: 모든 단계를 순차적으로 적용
df_ready = pipeline.fit(df).transform(df)

# 4. ML 모델 학습용 데이터 준비 (필요한 컬럼만 선택)
# - scaled_features: 표준화된 피처 벡터
# - Purchased: 타겟 변수 (예측할 값, 0 또는 1)
# - 컬럼명을 "features"로 변경 (PySpark ML 모델의 기본 입력 컬럼명)
data = df_ready.select("scaled_features", "Purchased") \
    .withColumnRenamed("scaled_features", "features")

# 5. 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=SEED)
train_data.count(), test_data.count()

# %%
# 1. 랜덤 포레스트 분류 모델 생성
rf = RandomForestClassifier(
    featuresCol="features",    # 입력 피처 컬럼명
    labelCol="Purchased",      # 타겟 변수 (예측할 값, 0 또는 1)
    seed=SEED                  # 랜덤 시드 고정 (재현 가능한 결과)
)

# 2. 학습 데이터로 모델 학습
rf_model = rf.fit(train_data)

# 3. 테스트 데이터로 예측 수행
# 결과: 원본 컬럼 + rawPrediction + probability + prediction 추가
rf_preds = rf_model.transform(test_data)

# %%
# 그래디언트 부스팅 트리 분류 모델 생성
gbt = GBTClassifier(featuresCol="features", labelCol="Purchased", seed=SEED)
gbt_model = gbt.fit(train_data)
gbt_preds = gbt_model.transform(test_data)

# %%
# 1. 평가 지표 설정
# AUC-ROC 평가기 (이진 분류 성능)
auc_eval = BinaryClassificationEvaluator(
    labelCol="Purchased",              # 실제 라벨 (정답)
    rawPredictionCol="rawPrediction",  # 모델의 원시 예측값
    metricName="areaUnderROC"          # 평가 지표: AUC-ROC (0~1, 높을수록 좋음)
)

# 정확도 평가기
acc_eval = MulticlassClassificationEvaluator(
    labelCol="Purchased",              # 실제 라벨 (정답)
    predictionCol="prediction",        # 모델의 최종 예측값 (0 또는 1)
    metricName="accuracy"              # 평가 지표: 정확도 (맞춘 비율)
)

# 2. 랜덤 포레스트 모델 성능 평가
rf_auc = auc_eval.evaluate(rf_preds)   # RF의 AUC-ROC 계산
rf_acc = acc_eval.evaluate(rf_preds)   # RF의 정확도 계산

# 3. 그래디언트 부스팅 트리 모델 성능 평가
gbt_auc = auc_eval.evaluate(gbt_preds) # GBT의 AUC-ROC 계산
gbt_acc = acc_eval.evaluate(gbt_preds) # GBT의 정확도 계산

# 4. 결과 출력 (두 모델 비교)
print("RF  - AUC:", rf_auc, " Accuracy:", rf_acc)
print("GBT - AUC:", gbt_auc, " Accuracy:", gbt_acc)

# %%
# 랜덤 포레스트 모델의 피처 중요도 출력
# 값이 클수록 해당 피처가 중요함 (합계 = 1.0)
print("RF feature importances:", rf_model.featureImportances)

# %%
print("=== 모델별 성능 비교 (산출물에 기록) ===")
print("model,auc,accuracy")
print(f"RandomForestClassifier,{rf_auc:.4f},{rf_acc:.4f}")
print(f"GBTClassifier,{gbt_auc:.4f},{gbt_acc:.4f}")

spark.stop()
