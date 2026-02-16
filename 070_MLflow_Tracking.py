# %% [markdown]
# # MLflow 기반 실험 추적
# - 머신러닝 실험을 추적하고 관리하는 Databricks에서 만든 오픈소스 플랫폼  
# - ML 프로젝트의 전체 생명주기를 관리: run 생성, param/metric 기록, 모델 artifact 저장, 실험 비교  

# %%
# MLflow 설치
# !pip install -q mlflow

# %%
import os
import sys
import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

IN_COLAB = "google.colab" in sys.modules
BASE = "/content" if IN_COLAB else os.getcwd()
CSV_PATH = os.path.join(BASE, "Social_Network_Ads.csv")

# MLflow 실험 결과 저장 디렉토리
MLFLOW_DIR = os.path.join(BASE, "mlruns")
SEED = 42

# MLflow 추적 URI 설정 (로컬 파일 시스템 사용)
mlflow.set_tracking_uri("file://" + os.path.abspath(MLFLOW_DIR))
# 실험 결과를 mlruns 폴더에 저장

# MLflow 실험 이름 설정
mlflow.set_experiment("ncs_spark")
# 실험 이름: "ncs_spark_day1" (여러 run을 그룹화)

spark = SparkSession.builder.appName("MLflow").getOrCreate()

# %%
# 1. CSV 파일을 DataFrame으로 읽기
df = (spark.read.format("csv")
    .option("header", "true")            # 첫 번째 행을 헤더로 사용
    .option("inferSchema", "true")    # 데이터 타입 자동 추론
    .load(CSV_PATH))

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

# 3. 전처리 파이프라인 생성 및 적용 - 모든 전처리 단계를 순서대로 묶음
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])

# 파이프라인 학습 후 데이터 변환
df_ready = pipeline.fit(df).transform(df)

# 4. ML 모델 학습용 데이터 준비 (피처와 라벨만 선택)
data = df_ready.select("scaled_features", "Purchased") \
    .withColumnRenamed("scaled_features", "features")

# 5. 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
train_data, test_data = data.randomSplit([0.8, 0.2], seed=SEED)

train_data.count(), test_data.count()

# %% [markdown]
# Run 1: Baseline LR

# %%
# MLflow run 시작 (실험 기록 시작)
# with 블록이 끝나면 run이 자동으로 종료되고 모든 정보가 저장됨

with mlflow.start_run(run_name="baseline_lr"):
    # run_name: 이 실험의 이름 (UI에서 식별용)

    # 1. 로지스틱 회귀 모델 생성
    lr = LogisticRegression(
        featuresCol="features",    # 입력 피처 컬럼
        labelCol="Purchased"       # 타겟 변수
    )

    # 2. 학습 데이터로 모델 학습
    model = lr.fit(train_data)

    # 3. 테스트 데이터로 예측
    preds = model.transform(test_data)

    # 4. 성능 평가 (AUC-ROC 계산)
    auc = BinaryClassificationEvaluator(
        labelCol="Purchased",              # 실제 라벨
        rawPredictionCol="rawPrediction",  # 모델의 원시 예측값
        metricName="areaUnderROC"          # 평가 지표: AUC-ROC
    ).evaluate(preds)

    # 5. MLflow에 파라미터 기록
    mlflow.log_param("model", "LogisticRegression")      # 모델 종류
    mlflow.log_param("regParam", str(lr.getRegParam()))  # 정규화 파라미터 값

    # 6. MLflow에 성능 지표 기록
    mlflow.log_metric("test_auc", auc)  # 테스트 AUC 값

    # 7. MLflow에 모델 저장
    mlflow.spark.log_model(model, "model")
    # 학습된 모델을 "model"이라는 이름으로 저장
    # 나중에 다운로드하거나 재사용 가능

# %% [markdown]
# Run 2: Tuned LR (CrossValidator best model)

# %%
# 1. 로지스틱 회귀 모델 생성
lr = LogisticRegression(
    featuresCol="features",    # 입력 피처 컬럼
    labelCol="Purchased"       # 타겟 변수
)

# 2. 평가 지표 설정 (AUC-ROC)
evaluator = BinaryClassificationEvaluator(
    labelCol="Purchased",              # 실제 라벨
    rawPredictionCol="rawPrediction",  # 모델의 원시 예측값
    metricName="areaUnderROC"          # 평가 지표: AUC-ROC
)

# 3. 하이퍼파라미터 그리드 생성 (테스트할 파라미터 조합)
# 총 2 × 2 = 4가지 조합
param_grid = (ParamGridBuilder()
    .addGrid(lr.regParam, [0.01, 0.1])          # 정규화 강도 (2개 값)
    .addGrid(lr.elasticNetParam, [0.0, 0.5])     # L1/L2 정규화 비율 (2개 값)
    .build())

# 4. 교차검증 설정 (최적 파라미터 찾기)
# 총 학습 횟수: 4가지 조합 × 3 Folds = 12번
cv = CrossValidator(
    estimator=lr,                      # 학습할 모델
    estimatorParamMaps=param_grid,     # 테스트할 파라미터 조합들
    evaluator=evaluator,               # 성능 평가 지표
    numFolds=3,                        # 3-Fold 교차검증
    seed=SEED                          # 랜덤 시드 고정
)

# 5. MLflow run 시작 (튜닝된 모델 실험 기록)
with mlflow.start_run(run_name="tuned_lr"):
    # run_name: 이 실험의 이름 (baseline_lr과 구분)

    # 4가지 조합을 3-Fold CV로 평가하여 최고 성능 조합 선택
    cv_model = cv.fit(train_data)

    # 가장 높은 평균 AUC를 가진 모델 추출
    best = cv_model.bestModel

    # 테스트 데이터로 예측
    preds = cv_model.transform(test_data)

    # 테스트 성능 평가
    auc_tuned = evaluator.evaluate(preds)

    # MLflow에 파라미터 기록
    mlflow.log_param("model", "LogisticRegression")                  # 모델 종류
    mlflow.log_param("regParam", str(best.getRegParam()))            # 최적 정규화 강도
    mlflow.log_param("elasticNetParam", str(best.getElasticNetParam()))  # 최적 L1/L2 비율

    # MLflow에 성능 지표 기록 - 튜닝 후 테스트 AUC (baseline_lr의 AUC와 비교 가능)
    mlflow.log_metric("test_auc", auc_tuned)

    # MLflow에 최적 모델 저장 -  최고 성능 모델을 저장 (나중에 재사용 가능)
    mlflow.spark.log_model(best, "model")

# %% [markdown]
# 실험 비교: MLflow UI에서 mlruns 폴더를 tracking_uri로 열어 run 목록 확인. 산출물 템플릿에 run_id, metric 기록.

# %%
# MLflow 실험 결과 저장 위치 출력
print("MLflow runs saved under:", MLFLOW_DIR)

# # "ncs_spark"라는 이름의 실험 객체 가져오기
experiment = mlflow.get_experiment_by_name("ncs_spark")

# 해당 실험의 모든 run(실행 기록) 검색
# 실험 ID를 사용하여 모든 run의 정보를 DataFrame으로 가져옴
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# 주요 컬럼만 선택하여 출력 (성능 비교용)
runs[["run_id", "metrics.test_auc", "params.regParam", "params.elasticNetParam"]]
# - run_id: 각 실험의 고유 ID
# - metrics.test_auc: 테스트 데이터의 AUC 성능 지표
# - params.regParam: 정규화 강도 파라미터
# - params.elasticNetParam: L1/L2 정규화 비율 파라미터

# %%
spark.stop()
