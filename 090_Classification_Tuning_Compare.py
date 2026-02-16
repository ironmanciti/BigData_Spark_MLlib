# %% [markdown]
# # 분류 모델 튜닝 및 성능 비교
# - RF/GBT 튜닝, MLflow 비교
# - 산출물: 최종 분류 모델 선정 보고서

# %%
# MLflow 설치
# !pip install -q mlflow

# %%
import os
import sys
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

IN_COLAB = "google.colab" in sys.modules
BASE = "/content" if IN_COLAB else os.getcwd()
CSV_PATH = os.path.join(BASE, "Social_Network_Ads.csv")
MLFLOW_DIR = os.path.join(BASE, "mlruns")
SEED = 42

mlflow.set_tracking_uri("file://" + os.path.abspath(MLFLOW_DIR))
mlflow.set_experiment("ncs_spark_classification")

spark = SparkSession.builder.appName("Classification_Tuning").getOrCreate()

# %%
# 1. CSV 파일을 DataFrame으로 읽기
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(CSV_PATH)

# 2. 전처리 단계 정의
# StringIndexer: 문자열을 숫자로 변환 (Gender → Gender_idx)
indexer = StringIndexer(inputCol="Gender", outputCol="Gender_idx") \
    .setHandleInvalid("keep")  # 학습에 없던 값 처리 (에러 방지)

# OneHotEncoder: 범주형 숫자를 원-핫 벡터로 변환 (Gender_idx → Gender_ohe)
encoder = OneHotEncoder(inputCols=["Gender_idx"], outputCols=["Gender_ohe"])

# VectorAssembler: 여러 피처를 하나의 벡터로 결합
assembler = VectorAssembler(
    inputCols=["Age", "EstimatedSalary", "Gender_ohe"],  # 입력: 나이, 연봉, 성별(원-핫)
    outputCol="features"                                  # 출력: features 벡터
)

# StandardScaler: 피처 표준화 (평균 0, 분산 1)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# 3. 전처리 파이프라인 생성 및 적용
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])

# 파이프라인 학습 후 데이터 변환
df_ready = pipeline.fit(df).transform(df)

# 4. ML 모델 학습용 데이터 준비 (필요한 컬럼만 선택)
data = df_ready.select("scaled_features", "Purchased") \
    .withColumnRenamed("scaled_features", "features")

# 5. 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=SEED)

# 6. 평가 지표 설정 (AUC-ROC)
evaluator = BinaryClassificationEvaluator(
    labelCol="Purchased",              # 실제 라벨 (정답, 0 또는 1)
    rawPredictionCol="rawPrediction",  # 모델의 원시 예측값 (확률 변환 전)
    metricName="areaUnderROC"          # 평가 지표: AUC-ROC (0~1, 높을수록 좋음)
)

# %%
# 1. 랜덤 포레스트 분류 모델 생성
rf = RandomForestClassifier(
    featuresCol="features",    # 입력 피처 컬럼명
    labelCol="Purchased",      # 타겟 변수 (예측할 값)
    seed=SEED                     # 랜덤 시드 고정 (재현성)
)

# 2. 하이퍼파라미터 그리드 생성 (테스트할 파라미터 조합)
rf_grid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [20, 50])    # 트리 개수 (2개 값)
    .addGrid(rf.maxDepth, [3, 5])       # 트리 최대 깊이 (2개 값)
    .build())
# 총 2 × 2 = 4가지 조합

# 3. 교차검증 설정 (최적 파라미터 찾기)
rf_cv = CrossValidator(
    estimator=rf,                      # 학습할 모델 (랜덤 포레스트)
    estimatorParamMaps=rf_grid,        # 테스트할 파라미터 조합들
    evaluator=evaluator,               # 성능 평가 지표 (AUC-ROC)
    numFolds=3,                        # 3-Fold 교차검증
    seed=SEED                          # 랜덤 시드 고정
)
# 총 학습 횟수: 4가지 조합 × 3 Folds = 12번

# 4. MLflow run 시작 (튜닝된 랜덤 포레스트 실험 기록)
with mlflow.start_run(run_name="rf_tuned"):

    # 교차검증으로 최적 모델 학습
    rf_cv_model = rf_cv.fit(train_data)
    # 4가지 조합을 3-Fold CV로 평가하여 최고 성능 조합 선택

    # 최적 모델을 사용하여 예측 수행
    rf_preds = rf_cv_model.transform(test_data)

    # 테스트 성능 평가 (AUC-ROC)
    rf_auc = evaluator.evaluate(rf_preds)

    # MLflow에 파라미터 기록
    mlflow.log_param("model", "RandomForestClassifier")                        # 모델 종류
    mlflow.log_param("numTrees", str(rf_cv_model.bestModel.getNumTrees))    # 최적 트리 개수
    mlflow.log_param("maxDepth", str(rf_cv_model.bestModel.getMaxDepth))    # 최적 트리 깊이

    # MLflow에 성능 지표 기록
    mlflow.log_metric("test_auc", rf_auc)

    # MLflow에 최적 모델 저장
    mlflow.spark.log_model(rf_cv_model.bestModel, "model")

# %%
gbt = GBTClassifier(featuresCol="features", labelCol="Purchased", seed=SEED)
gbt_grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3, 5]).addGrid(gbt.maxIter, [20, 50]).build()
gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_grid, evaluator=evaluator, numFolds=3, seed=SEED)

with mlflow.start_run(run_name="gbt_tuned"):
    gbt_cv_model = gbt_cv.fit(train_data)
    gbt_preds = gbt_cv_model.transform(test_data)
    gbt_auc = evaluator.evaluate(gbt_preds)
    mlflow.log_param("model", "GBTClassifier")
    mlflow.log_param("maxDepth", str(gbt_cv_model.bestModel.getMaxDepth()))
    mlflow.log_param("maxIter", str(gbt_cv_model.bestModel.getMaxIter()))
    mlflow.log_metric("test_auc", gbt_auc)
    mlflow.spark.log_model(gbt_cv_model.bestModel, "model")

# %%
print("RF tuned AUC:", rf_auc)
print("GBT tuned AUC:", gbt_auc)
print("최종 분류 모델 선정: AUC가 높은 모델을 선택")

spark.stop()
