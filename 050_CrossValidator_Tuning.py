# %% [markdown]
# #  교차검증 및 하이퍼파라미터 튜닝
# - ParamGridBuilder, CrossValidator, BestModel 선택
# - 산출물: 튜닝 전/후 성능 비교표, 최적 파라미터 기록

# %%
import os
import sys
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession

IN_COLAB = "google.colab" in sys.modules
BASE = "/content" if IN_COLAB else os.getcwd()
CSV_PATH = os.path.join(BASE, "Social_Network_Ads.csv")
SEED = 42

spark = SparkSession.builder.appName("CrossValidator").getOrCreate()

# %%
df = (spark.read.format("csv")
    .option("header", "true")           # 첫 번째 행을 헤더로 사용
    .option("inferSchema", "true")   # 데이터 타입 자동 추론
    .load(CSV_PATH))

# 전처리 단계 정의
indexer = StringIndexer(inputCol="Gender", outputCol="Gender_idx") \
    .setHandleInvalid("keep")  # 문자열을 숫자로 변환 (Male→0, Female→1)

# 숫자를 원-핫 벡터로 변환 (0→[1,0], 1→[0,1])
encoder = OneHotEncoder(inputCols=["Gender_idx"], outputCols=["Gender_ohe"])

# 여러 피처를 하나의 벡터로 결합
assembler = VectorAssembler(
    inputCols=["Age", "EstimatedSalary", "Gender_ohe"],
    outputCol="features"
)

# 피처를 표준화 (평균 0, 분산 1)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Pipeline 생성 및 적용 - 전처리 단계들을 순서대로 묶음
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])

# 파이프라인 학습 후 데이터 변환
df_ready = pipeline.fit(df).transform(df)

# ML 모델 학습용 데이터 준비 (피처와 라벨만 선택)
data = df_ready.select("scaled_features", "Purchased") \
    .withColumnRenamed("scaled_features", "features")  # scaled_features를 features로 이름 변경

# 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=SEED)

# %%
# 1. 로지스틱 회귀 모델 생성
lr = LogisticRegression(
    featuresCol="features",    # 입력 피처 컬럼명
    labelCol="Purchased"       # 타겟 변수 (예측할 값, 0 또는 1)
)

# 2. 평가 지표 설정 (AUC-ROC)
evaluator = BinaryClassificationEvaluator(
    labelCol="Purchased",                      # 실제 라벨
    rawPredictionCol="rawPrediction",    # 모델의 원시 예측값
    metricName="areaUnderROC"          # 평가 지표: AUC-ROC (높을수록 좋음)
)

# 3. 하이퍼파라미터 그리드 생성 (테스트할 파라미터 조합)
param_grid = (ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])            # 정규화 강도 (2개 값)
    .addGrid(lr.elasticNetParam, [0.0, 0.5])    # L1/L2 정규화 비율 (2개 값)
    .build())
# 총 2 × 2 = 4가지 조합 테스트

# 4. 교차검증 설정 (최적 파라미터 찾기)
cv = CrossValidator(
    estimator=lr,                                    # 학습할 모델 (로지스틱 회귀)
    estimatorParamMaps=param_grid,     # 테스트할 파라미터 조합들
    evaluator=evaluator,               # 성능 평가 지표 (AUC-ROC)
    numFolds=3,                           # 3-Fold 교차검증 (데이터를 3등분)
    seed=SEED,                           # 랜덤 시드 고정 (재현 가능)
)
# 총 학습 횟수: 4가지 조합 × 3 Folds = 12번 학습

# %%
# 1. 교차검증으로 최적 모델 학습
cv_model = cv.fit(train_data)
# 내부 동작:
# - 4가지 파라미터 조합 × 3-Fold = 총 12번 학습
# - 각 조합의 평균 AUC 계산
# - 가장 높은 AUC를 가진 조합 선택

# 2. 최적 모델 추출
best_lr = cv_model.bestModel
# 교차검증 결과 가장 성능이 좋았던 모델
# (최적 하이퍼파라미터가 적용된 로지스틱 회귀 모델)

# 3. 테스트 데이터로 예측 수행
predictions = cv_model.transform(test_data)
# 최적 모델을 사용하여 테스트 데이터 예측
# 결과: 원본 컬럼 + rawPrediction + probability + prediction

# 4. 튜닝된 모델의 성능 평가 (AUC-ROC)
auc_tuned = evaluator.evaluate(predictions)

# %%
print("Best model params:")
print(f"  regParam: {best_lr.getRegParam()}")
print(f"  elasticNetParam: {best_lr.getElasticNetParam()}")
print(f"  Test AUC: {auc_tuned:.4f}")

# %%
spark.stop()
