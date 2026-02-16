# %% [markdown]
# Day2-5교시: 모델 저장·로딩·버전 관리
# - PipelineModel.save / load, 경로 전략
# - 산출물: 모델 디렉토리 구조, 버전 기록표

# %%
import os
import sys
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql import SparkSession

IN_COLAB = "google.colab" in sys.modules
BASE = "/content" if IN_COLAB else os.getcwd()
CSV_PATH = os.path.join(BASE, "TestData", "Social_Network_Ads.csv")
MODEL_DIR = os.path.join(BASE, "saved_models")
SEED = 42

spark = SparkSession.builder.appName("Day2_Model_SaveLoad").getOrCreate()

# %%
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(CSV_PATH)
indexer = StringIndexer(inputCol="Gender", outputCol="Gender_idx").setHandleInvalid("keep")
encoder = OneHotEncoder(inputCols=["Gender_idx"], outputCols=["Gender_ohe"])
assembler = VectorAssembler(inputCols=["Age", "EstimatedSalary", "Gender_ohe"], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
lr = LogisticRegression(featuresCol="scaled_features", labelCol="Purchased")
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, lr])

# %%
model = pipeline.fit(df)
path_v1 = os.path.join(MODEL_DIR, "lr_pipeline_v1")
model.write().overwrite().save(path_v1)
print("Saved to", path_v1)

# %%
from pyspark.ml.pipeline import PipelineModel
loaded = PipelineModel.load(path_v1)
predictions = loaded.transform(df.limit(10))
predictions.select("Age", "EstimatedSalary", "prediction").show()

# %%
path_v2 = os.path.join(MODEL_DIR, "lr_pipeline_v2")
model.write().overwrite().save(path_v2)
print("버전 전략: saved_models/lr_pipeline_v1, lr_pipeline_v2 등으로 구분. 배포 시점·파라미터를 버전 기록표에 남기세요.")

spark.stop()
