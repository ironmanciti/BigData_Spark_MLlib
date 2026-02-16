# %% [markdown]
# # Spark 아키텍처 & DataFrame 실무
# - Spark 실행 구조 (Driver/Executor), Lazy Evaluation
# - DataFrame 중심 처리, CSV vs Parquet
# - 산출물: 스키마 정의 코드

# %%
# Colab: TestData 폴더를 /content/에 업로드하세요.
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# Colab 환경인지 확인
IN_COLAB = "google.colab" in sys.modules
# Colab이면 /content, 아니면 현재 작업 디렉토리를 BASE로 설정
BASE = "/content" if IN_COLAB else os.getcwd()

# CSV 파일 경로 설정
CSV_PATH = os.path.join(BASE, "titanic.csv")
# 출력할 Parquet 파일 경로 설정
OUTPUT_PARQUET = os.path.join(BASE, "out_titanic.parquet")

# SparkSession 생성 (DataFrame 및 Parquet 처리를 위한 애플리케이션)
spark = SparkSession.builder.appName("DataFrame_Parquet").getOrCreate()

# %% [markdown]
# ### 스키마 명시 정의

# %%
# Titanic 데이터셋의 스키마 정의
titanic_schema = StructType([
    StructField("PassengerId", IntegerType(), True),  # 승객 ID (정수형, Null 허용)
    StructField("Survived", IntegerType(), True),     # 생존 여부 (0: 사망, 1: 생존)
    StructField("Pclass", IntegerType(), True),       # 객실 등급 (1, 2, 3등석)
    StructField("Name", StringType(), True),          # 승객 이름
    StructField("Sex", StringType(), True),           # 성별
    StructField("Age", DoubleType(), True),           # 나이 (실수형)
    StructField("SibSp", IntegerType(), True),        # 함께 탑승한 형제자매/배우자 수
    StructField("Parch", IntegerType(), True),        # 함께 탑승한 부모/자녀 수
    StructField("Ticket", StringType(), True),        # 티켓 번호
    StructField("Fare", DoubleType(), True),          # 요금 (실수형)
    StructField("Cabin", StringType(), True),         # 객실 번호
    StructField("Embarked", StringType(), True),      # 탑승 항구 (C, Q, S)
])

# %% [markdown]
# CSV 읽기 (스키마 명시)

# %%
# CSV 파일을 DataFrame으로 읽기
df_csv = (spark.read.format("csv")  # CSV 형식 지정
    .option("header", "true")        # 첫 번째 행을 헤더로 사용
    .option("sep", ",")              # 구분자를 쉼표로 설정
    .schema(titanic_schema)          # 미리 정의된 스키마 적용
    .load(CSV_PATH))                 # CSV 파일 로드

# 상위 5개 행 출력
df_csv.limit(5).show()

# %% [markdown]
# DataFrame 기본 작업: 스키마, 데이터 타입, 통계

# %%
# 스키마 확인
df_csv.printSchema()

# %%
# 컬럼명과 데이터 타입 (Python 리스트)
df_csv.dtypes

# %%
# 수치형 요약 통계 (count, mean, stddev, min, max)
df_csv.describe("Age", "Fare").show()

# %%
# 요약 통계 (count, mean, stddev, min, 25%, 50%, 75%, max)
df_csv.summary().toPandas()

# %% [markdown]
# Data Filtering: 조건에 맞는 행만 선택

# %%
# Survived == 1 인 행만
df_csv.filter(F.col("Survived") == 1).limit(5).show()

# %%
# 복합 조건: Pclass가 1이고 Age가 30 이상 (and 사용)
df_csv.where((F.col("Pclass") == 1) & (F.col("Age") >= 30)).limit(5).show()

# %% [markdown]
# Grouping & Aggregation: 그룹별 집계

# %%
# Pclass(객실 등급)별 승객 수, 평균 요금, 최대 나이 집계
df_csv.groupBy("Pclass").agg(
    F.count("*").alias("cnt"),                 # 각 등급별 승객 수
    F.avg("Fare").alias("avg_fare"),     # 각 등급별 평균 요금
    F.max("Age").alias("max_age"),       # 각 등급별 최대 나이
).orderBy("Pclass").show()               # Pclass 기준 오름차순 정렬 후 출력

# %%
# Sex별 생존자 수
df_csv.filter(F.col("Survived") == 1).groupBy("Sex").count().orderBy(F.desc("count")).show()

# %% [markdown]
# 두 DataFrame의 Join

# %%
# Pclass 설명용 작은 테이블 생성
# 객실 등급 정보를 담은 DataFrame 생성 (등급 번호와 라벨 매핑)
class_info = spark.createDataFrame(
    [(1, "1st"), (2, "2nd"), (3, "3rd")],  # 데이터: (등급 번호, 등급 라벨) 튜플 리스트
    ["Pclass", "ClassLabel"]                # 컬럼명: Pclass, ClassLabel
)
class_info.toPandas()

# %% [markdown]
# Pclass 컬럼을 기준으로 두 DataFrame을 내부 조인 (양쪽에 모두 존재하는 데이터만 포함)

# %%
# df_csv와 class_info를 Pclass 기준으로 조인 (inner)
joined = df_csv.join(class_info, on="Pclass", how="inner")
joined.select("PassengerId", "Pclass", "ClassLabel", "Fare").limit(10).toPandas()

# %% [markdown]
# CSV → Parquet 변환 저장

# %%
# DataFrame을 Parquet 형식으로 저장 (기존 파일이 있으면 덮어쓰기)
df_csv.write.mode("overwrite").parquet(OUTPUT_PARQUET)

# %% [markdown]
# Parquet 읽기 (전체)

# %%
# Parquet 파일을 DataFrame으로 읽기
df_pq = spark.read.parquet(OUTPUT_PARQUET)

# 상위 5개 행 출력
df_pq.limit(5).show()

# %% [markdown]
# 컬럼 프루닝: 필요한 컬럼만 선택 후 읽기 (Parquet은 컬럼 단위 저장이라 유리)

# %%
# Parquet 파일에서 특정 컬럼만 선택하여 읽기 (컬럼 프루닝)
df_pq_pruned = spark.read.parquet(OUTPUT_PARQUET).select("Survived", "Pclass", "Sex", "Age", "Fare")

# 상위 5개 행 출력
df_pq_pruned.limit(5).show()
