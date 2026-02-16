# PowerPoint 교재 생성 프롬프트

> 아래 프롬프트를 PowerPoint Claude 플러그인에 입력하여 PPT를 자동 생성합니다.

---

## 프롬프트 시작

당신은 IT 교육용 프레젠테이션 전문가입니다. 아래 지시에 따라 **"머신러닝 활용 빅데이터 분석 과정"** PPT 교재를 약 **105장** 분량으로 생성해 주세요.

---

### 과정 정보

- **과정명:** 머신러닝 활용 빅데이터 분석 과정 (NCS 기반)
- **교육 기간:** 2일 (16시간)
- **대상:** Python 기초, Pandas, 기초 통계를 이수한 훈련생
- **핵심 도구:** PySpark, Spark MLlib, MLflow, Parquet
- **교육 목표:**
  1. Spark 환경에서 대규모 데이터를 로드·처리하고, ML 파이프라인을 구축할 수 있다
  2. 교차검증과 하이퍼파라미터 튜닝으로 성능을 개선하고, MLflow로 실험을 추적할 수 있다
  3. 분류/회귀 각각에서 Baseline → 개선 계획을 수립하고, 근거 기반 의사결정을 도출할 수 있다

---

### 디자인 지침

- **언어:** 모든 슬라이드 내용은 한국어 (코드 내 한국어 주석 유지, 변수명/API는 영문 그대로)
- **스타일:** 깔끔한 전문가 스타일, 일관된 컬러 스킴 (파란색/회색 계열)
- **코드 블록:** 모노스페이스 폰트, 배경색 있는 코드 박스, 한국어 주석 포함
- **이론 슬라이드:** 다이어그램 + 핵심 불릿 포인트 (3~5개)
- **실습 슬라이드:** 제목에 `[실습]` 접두사, 코드 블록 + 출력 예시
- **섹션 구분:** 각 섹션 끝에 "핵심 정리" 요약 슬라이드 포함
- **아이콘 사용:** 이론(책 아이콘), 코드(터미널 아이콘), 다이어그램(차트 아이콘), 팁(전구 아이콘)

---

### 전체 슬라이드 구조

```
[표지 & 과정 안내]  ..................  4장  (슬라이드 1~4)

━━ Day 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[섹션 1] 빅데이터 기초  ...............  8장  (슬라이드 5~12)
[섹션 2] Apache Spark 아키텍처  .......  10장 (슬라이드 13~22)
[섹션 3] DataFrame 실습  ..............  8장  (슬라이드 23~30)
[섹션 4] Spark SQL 이론+실습  .........  10장 (슬라이드 31~40)
[섹션 5] MLlib 파이프라인 이론  .......  8장  (슬라이드 41~48)
[섹션 6] 분류 베이스라인 실습  ........  7장  (슬라이드 49~55)

━━ Day 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[섹션 7] 교차검증 & 하이퍼파라미터 튜닝  6장  (슬라이드 56~61)
[섹션 8] 교차검증 실습  ...............  4장  (슬라이드 62~65)
[섹션 9] MLflow 이론+실습  ............  7장  (슬라이드 66~72)
[섹션 10] 앙상블 기법 이론  ...........  7장  (슬라이드 73~79)
[섹션 11] RF/GBT 분류 실습  ..........  5장  (슬라이드 80~84)
[섹션 12] 회귀 분석 이론  .............  5장  (슬라이드 85~89)
[섹션 13] 회귀 실습  ..................  6장  (슬라이드 90~95)
[섹션 14] 모델 저장 & 트레이드오프  ...  5장  (슬라이드 96~100)
[섹션 15] 팀 미니 프로젝트 & 마무리  ..  5장  (슬라이드 101~105)
                                       ─────
                                       합계  105장
```

---

## 슬라이드별 상세 지시

---

### 표지 & 과정 안내 (슬라이드 1~4)

**슬라이드 1: 표지**
- 제목: "머신러닝 활용 빅데이터 분석 과정"
- 부제: "PySpark & Spark MLlib 실무 2일 과정"
- 하단: NCS 기반 | 2일(16시간) 과정

**슬라이드 2: 과정 개요 및 학습 목표**
- 2일 전체 로드맵을 시각적 타임라인으로 표현
- 3대 핵심 역량 목표:
  1. Spark 데이터 처리 & SQL 활용
  2. MLlib 파이프라인 기반 분류/회귀 모델 구축
  3. 교차검증·튜닝·MLflow를 통한 실험 관리
- "End-to-End ML 워크플로우" 흐름도: 데이터 수집 → 전처리 → 학습 → 평가 → 튜닝 → 배포

**슬라이드 3: 실습 환경 안내**
- 실습 환경 옵션: Google Colab (권장) / 로컬 Anaconda + PySpark
- 필요 라이브러리: pyspark, mlflow, matplotlib, scikit-learn, pandas
- 프로젝트 파일 구조:
  ```
  BigData_Spark_MLlib/
  ├── Data/                          # 실습 데이터 파일
  │   ├── titanic.csv
  │   ├── Social_Network_Ads.csv
  │   ├── OnlineRetail.csv
  │   ├── ratings.csv
  │   └── ...
  ├── 010_Spark_DataFrame_Parquet.ipynb
  ├── 020_Spark_SQL_Basic.ipynb
  ├── 030_Spark_SQL_EDA.ipynb
  ├── 040_MLlib_Classification_Baseline.ipynb
  ├── 050_CrossValidator_Tuning.ipynb
  ├── 070_MLflow_Tracking.ipynb
  ├── 080_RF_GBT_Classification.ipynb
  ├── 090_Classification_Tuning_Compare.ipynb
  ├── 100_Regression_LR_RF_GBT.ipynb
  └── 120_Model_Save_Load_Versioning.ipynb
  ```

**슬라이드 4: 2일 커리큘럼 시간표**
- 표 형식 (Day / 교시 / 내용 / 실습 노트북):

| Day | 교시 | 내용 | 노트북 |
|-----|------|------|--------|
| 1 | 1~2 | 빅데이터 기초 & Spark 아키텍처 | - |
| 1 | 3 | Spark DataFrame & Parquet | 010 |
| 1 | 4~5 | Spark SQL & EDA | 020, 030 |
| 1 | 6~7 | MLlib 파이프라인 & 분류 베이스라인 | 040 |
| 1 | 8 | 교차검증 & 튜닝 이론 | - |
| 2 | 1 | 교차검증 실습 | 050 |
| 2 | 2~3 | MLflow 실험 추적 | 070 |
| 2 | 4~5 | 앙상블 분류 (RF/GBT) 튜닝 | 080, 090 |
| 2 | 6 | 회귀 분석 (LR/RF/GBT) | 100 |
| 2 | 7 | 모델 저장 & 트레이드오프 | 120 |
| 2 | 8 | 팀 미니 프로젝트 & 발표 | - |

---

### 섹션 1: 빅데이터 기초 (슬라이드 5~12)

**슬라이드 5: 빅데이터 정의와 5V**
- 5V 개념을 원형 다이어그램으로 표현:
  - Volume (규모): 테라~페타바이트 규모 데이터
  - Velocity (속도): 실시간 스트리밍 데이터
  - Variety (다양성): 정형/비정형/반정형 데이터
  - Veracity (정확성): 데이터 품질과 신뢰성
  - Value (가치): 데이터에서 비즈니스 가치 추출
- 핵심 메시지: "단일 머신으로 처리할 수 없는 규모의 데이터 → 분산 처리 필요"

**슬라이드 6: CRISP-DM 방법론**
- 6단계 순환 다이어그램:
  1. 비즈니스 이해
  2. 데이터 이해
  3. 데이터 준비
  4. 모델링
  5. 평가
  6. 배포
- 이 과정에서 다루는 범위를 색상으로 강조 (3~6단계)

**슬라이드 7: 데이터 아키텍처 개요**
- Data Lake vs Data Warehouse 비교표:

| 구분 | Data Lake | Data Warehouse |
|------|-----------|----------------|
| 데이터 형태 | Raw (비정형 포함) | 정제/구조화 |
| 스키마 | Schema-on-Read | Schema-on-Write |
| 용도 | 탐색, ML, 로그 분석 | BI, 리포팅 |
| 기술 예시 | HDFS, S3 | Redshift, BigQuery |

**슬라이드 8: 데이터 형식 — CSV vs Parquet**
- Row-based (CSV) vs Columnar (Parquet) 저장 방식 비교 다이어그램
- 비교표:

| 항목 | CSV | Parquet |
|------|-----|---------|
| 저장 방식 | 행 기반 | 열 기반 |
| 압축 효율 | 낮음 | 높음 (Snappy/GZIP) |
| 컬럼 프루닝 | 불가 | 가능 (필요한 컬럼만 읽기) |
| 스키마 내장 | 없음 | 있음 |
| 읽기 속도 | 전체 스캔 | 선택적 스캔 |

**슬라이드 9: Parquet 상세 — 왜 빅데이터에서 중요한가**
- 컬럼 프루닝(Column Pruning): 필요한 컬럼만 읽어 I/O 절감
- Predicate Pushdown: 조건절을 스토리지 레벨에서 필터링
- 스키마 내장: 별도 스키마 정의 불필요
- 실제 코드 예시 (010 노트북에서):
  ```python
  # Parquet에서 필요한 컬럼만 선택 (컬럼 프루닝)
  df_pq_pruned = spark.read.parquet(OUTPUT_PARQUET) \
      .select("Survived", "Pclass", "Sex", "Age", "Fare")
  ```

**슬라이드 10: 분산 처리의 등장**
- 단일 머신의 한계 → 분산 파일 시스템 (GFS/HDFS)
- MapReduce 패러다임: Map(분할 처리) → Shuffle → Reduce(집계)
- MapReduce의 한계: 디스크 기반 → 반복 연산에 느림

**슬라이드 11: Apache Spark의 등장**
- MapReduce 한계 극복: In-Memory 처리 (최대 100배 빠름)
- 통합 플랫폼: SQL, ML, Streaming, Graph를 하나로
- 핵심 메시지: "빅데이터 ML의 사실상 표준 엔진"

**슬라이드 12: 섹션 1 핵심 정리**
- 빅데이터 5V 정의
- CSV(행 기반) vs Parquet(열 기반): 빅데이터는 Parquet 권장
- 분산 처리: MapReduce → Spark (In-Memory)
- 다음 섹션 예고: "Spark가 어떤 구조로 동작하는지 알아봅니다"

---

### 섹션 2: Apache Spark 아키텍처 (슬라이드 13~22)

**슬라이드 13: Spark 핵심 특징**
- 4대 특징을 아이콘과 함께:
  1. 속도: In-Memory 처리, DAG 최적화
  2. 범용성: SQL, ML, Streaming, Graph 통합
  3. 확장성: 수천 노드까지 수평 확장
  4. 내결함성: RDD 계보(Lineage)로 자동 복구

**슬라이드 14: Spark 클러스터 아키텍처**
- Driver / Cluster Manager / Executor 3계층 다이어그램:
  - **Driver Program**: SparkSession 생성, 작업 계획, 결과 수집
  - **Cluster Manager**: 리소스 할당 (YARN, Mesos, Standalone)
  - **Executor**: 실제 데이터 처리, 캐시 저장
- 화살표로 통신 흐름 표시

**슬라이드 15: SparkSession — 모든 것의 시작점**
- SparkSession = Spark 애플리케이션의 진입점 (entry point)
- 코드 예시:
  ```python
  from pyspark.sql import SparkSession

  # SparkSession 생성
  spark = SparkSession.builder \
      .appName("DataFrame_Parquet") \
      .getOrCreate()
  ```
- SparkSession이 제공하는 기능: DataFrame API, SQL 엔진, Configuration

**슬라이드 16: RDD vs DataFrame vs Dataset**
- 3계층 비교표:

| 항목 | RDD | DataFrame | Dataset |
|------|-----|-----------|---------|
| 추상화 수준 | 낮음 | 높음 | 높음 |
| 최적화 | 수동 | Catalyst 자동 | Catalyst 자동 |
| 타입 안전성 | 있음 | 없음 | 있음 (JVM) |
| 사용 언어 | Python/Scala/Java | Python/Scala/Java/R | Scala/Java |

- 핵심 메시지: "PySpark에서는 **DataFrame** 중심으로 개발"

**슬라이드 17: Lazy Evaluation & DAG**
- Lazy Evaluation 개념:
  - Transformation (변환): filter, select, groupBy → 실행 계획만 기록
  - Action (실행): show, count, collect → 실제 실행 트리거
- DAG (Directed Acyclic Graph) 시각화:
  - 변환 체인이 DAG로 구성 → 최적화 후 실행
  - 장점: 불필요한 연산 제거, 파이프라이닝

**슬라이드 18: Spark 에코시스템**
- 5대 컴포넌트 다이어그램:
  - **Spark Core**: 기본 엔진 (RDD, Task Scheduling)
  - **Spark SQL**: 구조화 데이터 처리, DataFrame API
  - **MLlib**: 머신러닝 라이브러리
  - **Spark Streaming**: 실시간 스트리밍
  - **GraphX**: 그래프 처리
- 이 과정에서 다루는 부분 강조: Spark SQL + MLlib

**슬라이드 19: Catalyst Optimizer**
- 쿼리 최적화 파이프라인 흐름도:
  1. 논리 계획 (Unresolved Logical Plan)
  2. 분석 (Analysis) → Resolved Logical Plan
  3. 논리 최적화 (Logical Optimization) → Optimized Plan
  4. 물리 계획 생성 (Physical Planning)
  5. 코드 생성 (Code Generation) → 실행
- 핵심: "DataFrame API와 Spark SQL 모두 동일한 Catalyst로 최적화"

**슬라이드 20: Spark 실행 구조 상세**
- Job → Stage → Task 계층:
  - **Job**: Action 하나당 Job 하나
  - **Stage**: Shuffle 경계로 분리
  - **Task**: 파티션 하나당 Task 하나
- Shuffle 개념: 데이터 재분배 (groupBy, join 시 발생)

**슬라이드 21: 스키마 정의 (StructType)**
- 스키마란? 데이터의 구조(컬럼명, 타입, Null 허용) 정의
- 코드 예시 (010 노트북):
  ```python
  from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

  titanic_schema = StructType([
      StructField("PassengerId", IntegerType(), True),
      StructField("Survived", IntegerType(), True),
      StructField("Pclass", IntegerType(), True),
      StructField("Name", StringType(), True),
      StructField("Sex", StringType(), True),
      StructField("Age", DoubleType(), True),
      ...
  ])
  ```
- 주요 데이터 타입: IntegerType, StringType, DoubleType, TimestampType
- inferSchema vs 명시적 스키마 비교: 명시적 스키마가 더 안전하고 빠름

**슬라이드 22: 섹션 2 핵심 정리**
- Spark = In-Memory 분산 처리 엔진
- SparkSession → DataFrame → Lazy Evaluation → Action으로 실행
- Catalyst Optimizer가 자동 최적화
- 다음 섹션 예고: "직접 DataFrame을 만들어 봅니다"

---

### 섹션 3: DataFrame 실습 (슬라이드 23~30) — 노트북: 010_Spark_DataFrame_Parquet

**슬라이드 23: [실습] 010_Spark_DataFrame_Parquet 소개**
- 학습 목표:
  1. CSV 파일을 명시적 스키마로 읽기
  2. DataFrame 기본 작업: 스키마 확인, 통계, 필터링, 집계, 조인
  3. CSV → Parquet 변환 및 컬럼 프루닝 체험
- 데이터셋: `titanic.csv` (891건, 12개 컬럼)
- 환경 설정 코드:
  ```python
  BASE = "/content" if IN_COLAB else os.getcwd()
  CSV_PATH = os.path.join(BASE, "titanic.csv")
  spark = SparkSession.builder.appName("DataFrame_Parquet").getOrCreate()
  ```

**슬라이드 24: CSV 읽기 — 스키마 명시**
- 코드:
  ```python
  df_csv = (spark.read.format("csv")
      .option("header", "true")       # 헤더 사용
      .option("sep", ",")             # 쉼표 구분자
      .schema(titanic_schema)         # 명시적 스키마 적용
      .load(CSV_PATH))

  df_csv.limit(5).show()
  ```
- 핵심 포인트:
  - `format("csv")`: 파일 형식 지정
  - `schema()`: 미리 정의한 스키마 적용 (inferSchema보다 안전)
  - `load()`: 파일 경로 지정

**슬라이드 25: DataFrame 기본 작업**
- 스키마/통계 확인 코드:
  ```python
  df_csv.printSchema()                     # 스키마 출력
  df_csv.dtypes                            # 컬럼명 + 타입 리스트
  df_csv.describe("Age", "Fare").show()    # 수치형 요약 통계
  df_csv.summary().toPandas()              # 전체 요약 (25/50/75%)
  ```
- 각 메서드의 역할과 출력 형태 설명

**슬라이드 26: 필터링과 조건 검색**
- 코드:
  ```python
  # 단일 조건: 생존자만
  df_csv.filter(F.col("Survived") == 1).limit(5).show()

  # 복합 조건: 1등석 & 30세 이상
  df_csv.where(
      (F.col("Pclass") == 1) & (F.col("Age") >= 30)
  ).limit(5).show()
  ```
- 핵심: `filter()` = `where()` (동일 기능), `F.col()` 사용, `&`(AND), `|`(OR)

**슬라이드 27: 그룹별 집계 (groupBy / agg)**
- 코드:
  ```python
  # 등급별 승객 수, 평균 요금, 최대 나이
  df_csv.groupBy("Pclass").agg(
      F.count("*").alias("cnt"),
      F.avg("Fare").alias("avg_fare"),
      F.max("Age").alias("max_age"),
  ).orderBy("Pclass").show()
  ```
- 결과 테이블 예시를 함께 표시
- 핵심 함수: `groupBy()`, `agg()`, `count()`, `avg()`, `max()`, `alias()`, `orderBy()`

**슬라이드 28: DataFrame Join**
- 코드:
  ```python
  # 등급 라벨 테이블 생성
  class_info = spark.createDataFrame(
      [(1, "1st"), (2, "2nd"), (3, "3rd")],
      ["Pclass", "ClassLabel"]
  )

  # Inner Join
  joined = df_csv.join(class_info, on="Pclass", how="inner")
  joined.select("PassengerId", "Pclass", "ClassLabel", "Fare") \
      .limit(10).toPandas()
  ```
- Join 유형 설명: inner, left, right, full (이 실습은 inner)

**슬라이드 29: CSV → Parquet 변환 & 컬럼 프루닝**
- 코드:
  ```python
  # Parquet으로 저장
  df_csv.write.mode("overwrite").parquet(OUTPUT_PARQUET)

  # Parquet 읽기
  df_pq = spark.read.parquet(OUTPUT_PARQUET)

  # 컬럼 프루닝: 필요한 컬럼만 선택
  df_pq_pruned = spark.read.parquet(OUTPUT_PARQUET) \
      .select("Survived", "Pclass", "Sex", "Age", "Fare")
  ```
- 핵심: Parquet은 열 기반이라 `select()`로 필요한 컬럼만 읽으면 I/O 크게 절감

**슬라이드 30: 실습 010 핵심 정리**
- 주요 API 요약표:

| API | 용도 |
|-----|------|
| `spark.read.format("csv").schema().load()` | CSV 읽기 (스키마 명시) |
| `printSchema()`, `describe()`, `summary()` | 데이터 탐색 |
| `filter()`, `where()` | 조건 필터링 |
| `groupBy().agg()` | 그룹별 집계 |
| `join(df, on, how)` | DataFrame 조인 |
| `write.parquet()` / `read.parquet()` | Parquet 저장/읽기 |
| `.select()` + Parquet | 컬럼 프루닝 |

---

### 섹션 4: Spark SQL 이론+실습 (슬라이드 31~40) — 노트북: 020, 030

**슬라이드 31: Spark SQL 개요**
- Spark SQL = SQL 인터페이스로 DataFrame을 쿼리
- DataFrame API vs SQL API: 동일한 Catalyst Optimizer로 최적화 → 성능 동일
- 사용 시나리오: SQL 친숙한 분석가, 복잡한 조인/집계 쿼리
- `spark.sql("SELECT ...")` 으로 SQL 문 실행

**슬라이드 32: [실습] 020_Spark_SQL_Basic 소개**
- 학습 목표:
  1. Parquet/CSV 파일로 테이블 생성
  2. SQL 문으로 데이터 조회·집계·조인
  3. Temporary View 활용
- 데이터셋: `people_10M1.parquet` (1,000만건), `babyNamesUSYOB-mostpopular.csv`

**슬라이드 33: 테이블 생성 (CREATE TABLE)**
- 코드:
  ```sql
  -- Parquet 파일로 테이블 생성
  CREATE TABLE IF NOT EXISTS People10M
  USING parquet
  OPTIONS (path '/content/people_10M1.parquet')

  -- CSV 파일로 테이블 생성
  CREATE TABLE IF NOT EXISTS ssaNames
  USING csv
  OPTIONS (
      path '/content/babyNamesUSYOB-mostpopular.csv',
      header 'true',
      inferSchema 'true'
  )
  ```
- 핵심: `USING` 으로 파일 형식 지정, `OPTIONS`로 파일 경로 및 옵션

**슬라이드 34: SELECT / WHERE / DESCRIBE**
- 코드:
  ```sql
  -- 스키마 확인
  DESCRIBE People10M

  -- 조건부 조회
  SELECT firstName, lastName, salary
  FROM People10M
  WHERE salary > 50000
  LIMIT 10
  ```
- 핵심: SQL 문법 그대로 사용, `DESCRIBE`로 스키마 확인

**슬라이드 35: 수치 계산과 별칭 (AS)**
- 코드:
  ```sql
  -- 연봉의 20%를 저축액으로 계산
  SELECT firstName, salary,
         ROUND(salary * 0.2, 2) AS savings
  FROM People10M
  WHERE YEAR(birthDate) >= 1990
  LIMIT 10
  ```
- 핵심 함수: `ROUND()`, `YEAR()`, `AS` 별칭

**슬라이드 36: Temporary View**
- 개념: 임시 뷰 = 쿼리 결과를 가상 테이블로 등록 (메모리에만 존재)
- 코드:
  ```sql
  CREATE OR REPLACE TEMPORARY VIEW HighEarners AS
  SELECT firstName, lastName, salary
  FROM People10M
  WHERE salary > 80000
  ```
- 활용: 복잡한 쿼리를 단계별로 분리, 재사용 가능

**슬라이드 37: GROUP BY / ORDER BY / AVG**
- 코드:
  ```sql
  SELECT gender, ROUND(AVG(salary), 2) AS avg_salary, COUNT(*) AS cnt
  FROM People10M
  GROUP BY gender
  ORDER BY avg_salary DESC
  ```
- 핵심: 집계 함수(AVG, COUNT, SUM), 정렬(ORDER BY), 소수점(ROUND)

**슬라이드 38: SQL JOIN**
- 코드:
  ```sql
  SELECT a.firstName, b.firstName AS baby_name, b.total
  FROM People10M a
  INNER JOIN ssaNames b
  ON a.firstName = b.firstName
  LIMIT 10
  ```
- Join 유형: INNER, LEFT OUTER, RIGHT OUTER, FULL OUTER

**슬라이드 39: [실습] 030_Spark_SQL_EDA 소개**
- 학습 목표:
  1. SQL로 EDA 수행 (집계, 변환, 시각화)
  2. CAST로 타입 변환, MONTH() 함수 활용
  3. matplotlib으로 히스토그램/파이 차트 생성
- 데이터셋: `ratings.csv` (2,000만건), `OnlineRetail.csv`, `countries_iso3166b.csv`

**슬라이드 40: EDA 핵심 패턴 (030 하이라이트)**
- 4가지 EDA 패턴:
  1. **CAST 변환**: `CAST(InvoiceDate AS timestamp)` → 시계열 분석 가능
  2. **샘플링 + 시각화**: `orderBy(rand()).limit(N)` → 1% 샘플 히스토그램
     ```python
     sample_df = spark.sql("SELECT ... ORDER BY rand() LIMIT 2000")
     plt.hist(sample_df.toPandas()["rating"], bins=5)
     ```
  3. **Temporary View + GROUP BY**: 월별 매출 집계 뷰 생성
  4. **JOIN + 시각화**: 국가코드 테이블 JOIN 후 파이 차트
     - "Others" 통합 기법: 소규모 국가를 "Others"로 묶어 파이 차트 가독성 향상

---

### 섹션 5: MLlib 파이프라인 이론 (슬라이드 41~48)

**슬라이드 41: MLlib 개요**
- Spark MLlib = Spark의 머신러닝 라이브러리
- 지원 알고리즘: 분류, 회귀, 클러스터링, 추천, 피처 변환
- 핵심 철학: **파이프라인 기반** ML 워크플로우
- `pyspark.ml` 패키지 사용 (구 `pyspark.mllib`는 deprecated)

**슬라이드 42: Transformer vs Estimator**
- 다이어그램:
  - **Transformer**: `transform(DataFrame) → DataFrame`
    - 입력 DataFrame을 변환하여 새로운 컬럼 추가
    - 예: VectorAssembler, OneHotEncoder
  - **Estimator**: `fit(DataFrame) → Model(Transformer)`
    - 데이터로부터 학습 후 Model (= Transformer) 생성
    - 예: StringIndexer, StandardScaler, LogisticRegression
- 흐름: Estimator.fit(df) → Model → Model.transform(df)

**슬라이드 43: Pipeline 구조**
- Pipeline = 여러 Transformer/Estimator를 순서대로 연결
- 다이어그램:
  ```
  Pipeline(stages=[stage1, stage2, stage3, stage4])
      │
      ▼ .fit(df)
  PipelineModel
      │
      ▼ .transform(df)
  변환된 DataFrame
  ```
- 장점:
  - 한 번의 `fit()`으로 모든 단계 학습
  - `transform()`으로 새 데이터에 동일 변환 적용
  - 재현 가능한 ML 워크플로우

**슬라이드 44: 피처 엔지니어링 전체 흐름**
- 4단계 파이프라인 흐름도 (실습 040 기반):
  ```
  원본 데이터
    │
    ▼ StringIndexer
  Gender → Gender_idx (Male→0, Female→1)
    │
    ▼ OneHotEncoder
  Gender_idx → Gender_ohe ([1,0] or [0,1])
    │
    ▼ VectorAssembler
  [Age, Salary, Gender_ohe] → features 벡터
    │
    ▼ StandardScaler
  features → scaled_features (평균0, 분산1)
  ```

**슬라이드 45: StringIndexer & OneHotEncoder**
- **StringIndexer**:
  - 범주형 문자열 → 숫자 인덱스 (빈도순)
  - 예: Male(200건) → 0, Female(150건) → 1
  - `setHandleInvalid("keep")`: 학습에 없던 값 처리
- **OneHotEncoder**:
  - 숫자 인덱스 → 희소 벡터 (원-핫 인코딩)
  - 예: 0 → (1, [0], [1.0]), 1 → (1, [], [])
- 코드:
  ```python
  indexer = StringIndexer(inputCol="Gender", outputCol="Gender_idx") \
      .setHandleInvalid("keep")
  encoder = OneHotEncoder(inputCols=["Gender_idx"], outputCols=["Gender_ohe"])
  ```

**슬라이드 46: VectorAssembler & StandardScaler**
- **VectorAssembler**:
  - 여러 컬럼을 하나의 벡터 컬럼으로 결합
  - ML 모델의 입력 형태: 단일 벡터 컬럼 필요
  - 예: Age=19, Salary=19000, Gender_ohe=[1,0] → [19.0, 19000.0, 1.0, 0.0]
- **StandardScaler**:
  - 각 피처를 평균 0, 분산 1로 정규화
  - 수식: z = (x - mean) / std
  - 왜 필요한가: 스케일이 다른 피처 간 공정한 비교 (Age: 0~100 vs Salary: 0~200000)
- 코드:
  ```python
  assembler = VectorAssembler(
      inputCols=["Age", "EstimatedSalary", "Gender_ohe"],
      outputCol="features"
  )
  scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
  ```

**슬라이드 47: Logistic Regression 이론**
- **시그모이드 함수**: sigma(z) = 1 / (1 + e^(-z))
  - 출력 범위: 0 ~ 1 (확률로 해석)
  - 결정 경계: 0.5 기준 (> 0.5 → 양성, <= 0.5 → 음성)
- 시그모이드 그래프 시각화
- 이진 분류: 두 클래스(0/1) 분류에 적합
- 핵심 하이퍼파라미터:
  - `regParam`: 정규화 강도 (과적합 방지)
  - `elasticNetParam`: L1/L2 정규화 비율 (0=L2, 1=L1)

**슬라이드 48: 평가 지표 — AUC, Confusion Matrix**
- **Confusion Matrix** 다이어그램 (2x2 표):

|  | 예측: 양성 | 예측: 음성 |
|--|----------|----------|
| 실제: 양성 | TP (True Positive) | FN (False Negative) |
| 실제: 음성 | FP (False Positive) | TN (True Negative) |

- **평가 지표**:
  - Accuracy = (TP+TN) / 전체
  - Precision = TP / (TP+FP) — 양성 예측의 정확도
  - Recall = TP / (TP+FN) — 실제 양성 중 맞춘 비율
  - AUC-ROC: ROC 곡선 아래 면적 (0~1, 높을수록 좋음)
  - AUC-PR: Precision-Recall 곡선 아래 면적
- ROC 곡선 그래프 시각화

---

### 섹션 6: 분류 베이스라인 실습 (슬라이드 49~55) — 노트북: 040_MLlib_Classification_Baseline

**슬라이드 49: [실습] 040_MLlib_Classification_Baseline 소개**
- 학습 목표:
  1. 전처리 파이프라인 구성 및 실행
  2. Logistic Regression 베이스라인 모델 구축
  3. AUC, Accuracy, Confusion Matrix 평가
- 데이터셋: `Social_Network_Ads.csv`
  - 컬럼: Age, Gender, EstimatedSalary, Purchased (타겟)
  - 400건, 이진 분류 (Purchased: 0 또는 1)

**슬라이드 50: 데이터 로드 및 확인**
- 코드:
  ```python
  df = (spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(CSV_PATH))

  df.limit(3).toPandas()
  df.printSchema()
  ```
- 출력 예시: Age(int), Gender(string), EstimatedSalary(int), Purchased(int)

**슬라이드 51: 파이프라인 구성 코드**
- 전체 파이프라인 코드:
  ```python
  # 1. StringIndexer: Gender → Gender_idx
  indexer = StringIndexer(inputCol="Gender", outputCol="Gender_idx") \
      .setHandleInvalid("keep")

  # 2. OneHotEncoder: Gender_idx → Gender_ohe
  encoder = OneHotEncoder(inputCols=["Gender_idx"], outputCols=["Gender_ohe"])

  # 3. VectorAssembler: 피처 결합
  assembler = VectorAssembler(
      inputCols=["Age", "EstimatedSalary", "Gender_ohe"],
      outputCol="features"
  )

  # 4. StandardScaler: 표준화
  scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

  # Pipeline 생성
  pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler])
  ```

**슬라이드 52: Pipeline fit / transform**
- 코드:
  ```python
  # Pipeline 학습 + 변환
  model = pipeline.fit(df)
  transformed = model.transform(df)

  # 변환 결과 확인
  transformed.select("Age", "EstimatedSalary", "Gender",
                     "scaled_features", "Purchased") \
      .limit(5).toPandas()
  ```
- 내부 동작 설명:
  1. `indexer.fit(df)` → Gender 고유값 학습
  2. `encoder` (Transformer) → 원-핫 변환
  3. `assembler` (Transformer) → 벡터 결합
  4. `scaler.fit()` → 평균/표준편차 학습 → 표준화

**슬라이드 53: Train/Test 분리 & LR 학습**
- 코드:
  ```python
  # 데이터 준비
  data = df_ready.select("scaled_features", "Purchased") \
      .withColumnRenamed("scaled_features", "features")

  # 80:20 분할
  train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

  # Logistic Regression 학습
  lr = LogisticRegression(featuresCol="features", labelCol="Purchased")
  lr_model = lr.fit(train_data)

  # 예측
  predictions = lr_model.transform(test_data)
  ```

**슬라이드 54: 평가 지표 계산**
- 코드:
  ```python
  # AUC-ROC
  auc_eval = BinaryClassificationEvaluator(
      labelCol="Purchased", metricName="areaUnderROC")
  auc = auc_eval.evaluate(predictions)

  # AUC-PR
  pr_eval = BinaryClassificationEvaluator(
      labelCol="Purchased", metricName="areaUnderPR")
  pr_area = pr_eval.evaluate(predictions)

  # Accuracy
  acc_eval = MulticlassClassificationEvaluator(
      labelCol="Purchased", metricName="accuracy")
  accuracy = acc_eval.evaluate(predictions)
  ```
- 결과 출력 예시:
  ```
  AUC (ROC):  0.89xx
  AUC (PR):   0.87xx
  Accuracy:   0.85xx
  ```

**슬라이드 55: Confusion Matrix & Day 1 정리**
- Confusion Matrix 코드:
  ```python
  predictions.groupBy("Purchased", "prediction") \
      .count().orderBy("Purchased", "prediction").toPandas()
  ```
- Day 1 전체 요약:
  1. 빅데이터 기초 & Spark 아키텍처
  2. DataFrame 연산 & Parquet
  3. Spark SQL & EDA
  4. MLlib 파이프라인 & 분류 베이스라인 (AUC, Accuracy)
- Day 2 예고: "베이스라인을 어떻게 개선할 수 있을까? → 교차검증, 앙상블, MLflow"

---

### 섹션 7: 교차검증 & 하이퍼파라미터 튜닝 이론 (슬라이드 56~61)

**슬라이드 56: Day 2 오프닝**
- Day 1 핵심 복습 (1분 요약):
  - Spark DataFrame/SQL → MLlib Pipeline → LR Baseline
- Day 2 목표:
  1. 교차검증 & 하이퍼파라미터 튜닝
  2. MLflow 실험 추적
  3. 앙상블 기법 (RF/GBT)
  4. 회귀 분석
  5. 모델 저장 & 팀 프로젝트

**슬라이드 57: 하이퍼파라미터란?**
- 모델 파라미터 vs 하이퍼파라미터 비교표:

| 구분 | 모델 파라미터 | 하이퍼파라미터 |
|------|-------------|---------------|
| 결정 방법 | 학습 데이터로 자동 결정 | 사람이 사전 설정 |
| 예시 | 가중치(weight), 편향(bias) | regParam, numTrees, maxDepth |
| 변경 시점 | 학습 중 | 학습 전 |

- 핵심: "하이퍼파라미터 선택이 모델 성능을 크게 좌우"

**슬라이드 58: k-Fold 교차검증**
- k-Fold CV 다이어그램 (k=3 예시):
  ```
  데이터 전체: [Fold1 | Fold2 | Fold3]

  반복 1: [검증 | 학습  | 학습 ] → 성능1
  반복 2: [학습  | 검증 | 학습 ] → 성능2
  반복 3: [학습  | 학습  | 검증] → 성능3

  최종 성능 = (성능1 + 성능2 + 성능3) / 3
  ```
- 장점:
  - 모든 데이터가 한 번씩 검증에 사용 → 편향 감소
  - 단일 split보다 신뢰성 높은 성능 추정
- Spark MLlib에서: `CrossValidator(numFolds=3)`

**슬라이드 59: Grid Search (격자 탐색)**
- ParamGridBuilder 개념:
  ```python
  param_grid = (ParamGridBuilder()
      .addGrid(lr.regParam, [0.01, 0.1])        # 2개 값
      .addGrid(lr.elasticNetParam, [0.0, 0.5])   # 2개 값
      .build())
  # 총 조합: 2 x 2 = 4가지
  ```
- 격자 시각화 (2x2 표):

| | elasticNet=0.0 | elasticNet=0.5 |
|--|---------------|----------------|
| regParam=0.01 | 조합1 | 조합2 |
| regParam=0.1 | 조합3 | 조합4 |

- 총 학습 횟수: 4조합 x 3Folds = **12번** 학습

**슬라이드 60: CrossValidator 전체 구조**
- 다이어그램:
  ```
  CrossValidator
  ├── estimator: LogisticRegression
  ├── estimatorParamMaps: [4가지 조합]
  ├── evaluator: BinaryClassificationEvaluator(AUC-ROC)
  └── numFolds: 3
      │
      ▼ cv.fit(train_data)
  CrossValidatorModel
      │
      ▼ .bestModel
  최적 파라미터가 적용된 LogisticRegression 모델
  ```

**슬라이드 61: 스케일링 & 불균형 처리**
- **스케일링 기법 비교**:
  - StandardScaler: z = (x - mean) / std → 평균 0, 분산 1
  - MinMaxScaler: z = (x - min) / (max - min) → 0~1 범위
  - 실습에서 사용: StandardScaler (노트북 040~120 전체)
- **클래스 불균형 처리** (개념 소개):
  - 문제: 양성 10%, 음성 90% → 모두 음성으로 예측해도 정확도 90%
  - 해결 방법:
    1. 오버샘플링/언더샘플링
    2. 클래스 가중치 (LogisticRegression의 `weightCol` 파라미터)
    3. 평가 지표 변경 (Accuracy 대신 AUC, F1-score)

---

### 섹션 8: 교차검증 실습 (슬라이드 62~65) — 노트북: 050_CrossValidator_Tuning

**슬라이드 62: [실습] 050_CrossValidator_Tuning 소개**
- 학습 목표: Day 1 베이스라인 LR의 성능을 교차검증+그리드 탐색으로 개선
- 데이터셋: `Social_Network_Ads.csv` (동일)
- 전처리 파이프라인: 040과 동일 (IndexerOneHotAssemblerScaler)

**슬라이드 63: ParamGridBuilder & CrossValidator 코드**
- 핵심 코드:
  ```python
  # 로지스틱 회귀 모델
  lr = LogisticRegression(featuresCol="features", labelCol="Purchased")

  # 평가 지표: AUC-ROC
  evaluator = BinaryClassificationEvaluator(
      labelCol="Purchased", metricName="areaUnderROC")

  # 하이퍼파라미터 그리드 (2x2 = 4조합)
  param_grid = (ParamGridBuilder()
      .addGrid(lr.regParam, [0.01, 0.1])
      .addGrid(lr.elasticNetParam, [0.0, 0.5])
      .build())

  # 교차검증 (3-Fold, 총 12회 학습)
  cv = CrossValidator(
      estimator=lr,
      estimatorParamMaps=param_grid,
      evaluator=evaluator,
      numFolds=3, seed=42)
  ```

**슬라이드 64: 최적 모델 결과**
- 코드:
  ```python
  cv_model = cv.fit(train_data)
  best_lr = cv_model.bestModel

  predictions = cv_model.transform(test_data)
  auc_tuned = evaluator.evaluate(predictions)

  print(f"regParam: {best_lr.getRegParam()}")
  print(f"elasticNetParam: {best_lr.getElasticNetParam()}")
  print(f"Test AUC: {auc_tuned:.4f}")
  ```
- 결과 비교표:

| 모델 | regParam | elasticNetParam | Test AUC |
|------|----------|----------------|----------|
| Baseline LR (040) | 기본값 | 기본값 | 0.89xx |
| Tuned LR (050) | 최적값 | 최적값 | 0.91xx |

**슬라이드 65: 실습 050 핵심 정리**
- ParamGridBuilder → 파라미터 조합 생성
- CrossValidator → k-Fold CV로 최적 조합 탐색
- bestModel → 최고 성능 모델 자동 선택
- 핵심: "체계적 탐색 > 수동 시행착오"

---

### 섹션 9: MLflow 이론+실습 (슬라이드 66~72) — 노트북: 070_MLflow_Tracking

**슬라이드 66: MLflow 개요 — 왜 실험 추적이 필요한가**
- 문제 상황: 여러 모델, 여러 파라미터 조합 → "어떤 설정이 최고였지?"
- MLflow = Databricks가 만든 오픈소스 ML 라이프사이클 관리 플랫폼
- 4대 구성 요소:
  1. **Tracking**: 파라미터, 지표, 모델 기록 (이 과정에서 사용)
  2. **Projects**: 재현 가능한 코드 패키징
  3. **Models**: 모델 배포 관리
  4. **Registry**: 모델 버전 관리

**슬라이드 67: MLflow Tracking 핵심 개념**
- 다이어그램:
  ```
  Experiment (실험 그룹)
  └── Run 1 (baseline_lr)
  │   ├── Parameters: {model: LR, regParam: 0.0}
  │   ├── Metrics: {test_auc: 0.89}
  │   └── Artifacts: {model/}
  └── Run 2 (tuned_lr)
      ├── Parameters: {model: LR, regParam: 0.01, elasticNet: 0.0}
      ├── Metrics: {test_auc: 0.91}
      └── Artifacts: {model/}
  ```
- 핵심 용어: Experiment, Run, Parameter, Metric, Artifact

**슬라이드 68: [실습] 070_MLflow_Tracking 소개**
- 학습 목표:
  1. MLflow 환경 설정
  2. Baseline LR과 Tuned LR 두 Run 기록
  3. 실험 결과 비교
- 설정 코드:
  ```python
  mlflow.set_tracking_uri("file://" + os.path.abspath(MLFLOW_DIR))
  mlflow.set_experiment("ncs_spark")
  ```

**슬라이드 69: Run 1 — Baseline LR 기록**
- 코드:
  ```python
  with mlflow.start_run(run_name="baseline_lr"):
      lr = LogisticRegression(featuresCol="features", labelCol="Purchased")
      model = lr.fit(train_data)
      preds = model.transform(test_data)

      auc = BinaryClassificationEvaluator(
          labelCol="Purchased", metricName="areaUnderROC"
      ).evaluate(preds)

      mlflow.log_param("model", "LogisticRegression")
      mlflow.log_param("regParam", str(lr.getRegParam()))
      mlflow.log_metric("test_auc", auc)
      mlflow.spark.log_model(model, "model")
  ```
- 핵심 API: `start_run()`, `log_param()`, `log_metric()`, `log_model()`

**슬라이드 70: Run 2 — Tuned LR (CrossValidator) 기록**
- 코드:
  ```python
  with mlflow.start_run(run_name="tuned_lr"):
      cv_model = cv.fit(train_data)
      best = cv_model.bestModel
      preds = cv_model.transform(test_data)
      auc_tuned = evaluator.evaluate(preds)

      mlflow.log_param("model", "LogisticRegression")
      mlflow.log_param("regParam", str(best.getRegParam()))
      mlflow.log_param("elasticNetParam", str(best.getElasticNetParam()))
      mlflow.log_metric("test_auc", auc_tuned)
      mlflow.spark.log_model(best, "model")
  ```

**슬라이드 71: 실험 비교**
- 코드:
  ```python
  experiment = mlflow.get_experiment_by_name("ncs_spark")
  runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
  runs[["run_id", "metrics.test_auc", "params.regParam", "params.elasticNetParam"]]
  ```
- 결과 테이블 예시:

| run_id | test_auc | regParam | elasticNetParam |
|--------|----------|----------|-----------------|
| abc123 | 0.89xx | 0.0 | - |
| def456 | 0.91xx | 0.01 | 0.0 |

**슬라이드 72: MLflow UI 안내**
- 터미널에서 실행: `mlflow ui --backend-store-uri file:///path/to/mlruns`
- UI 기능 설명:
  - Run 목록 보기
  - 지표 비교 차트
  - 파라미터-성능 상관관계
  - 모델 아티팩트 다운로드
- 핵심: "모든 실험을 체계적으로 기록하고 비교"

---

### 섹션 10: 앙상블 기법 이론 (슬라이드 73~79)

**슬라이드 73: 의사결정 트리 (Decision Tree)**
- 트리 구조 다이어그램:
  - 루트 노드 → 분할 조건 → 리프 노드 (예측)
  - 분할 기준: 지니 불순도 (Gini Impurity) 또는 엔트로피
- 장점: 해석 가능, 전처리 최소화
- 단점: 과적합 경향, 불안정 (데이터 변화에 민감)

**슬라이드 74: 앙상블 학습 개요**
- 앙상블 = 여러 약한 학습기를 결합하여 강한 학습기 생성
- 두 가지 전략:
  1. **Bagging** (Bootstrap Aggregating): 병렬 학습, 분산 감소 → Random Forest
  2. **Boosting**: 순차 학습, 편향 감소 → Gradient Boosted Trees
- 다이어그램: 단일 트리 vs 앙상블 비교

**슬라이드 75: Bagging — Random Forest**
- 알고리즘 다이어그램:
  1. 원본 데이터에서 부트스트랩 샘플 N개 생성 (중복 허용)
  2. 각 샘플로 독립적인 Decision Tree 학습
  3. 각 트리에서 랜덤 피처 서브셋 선택 (트리 다양성 확보)
  4. 최종 예측: 다수결 투표 (분류) / 평균 (회귀)
- 주요 하이퍼파라미터:
  - `numTrees`: 트리 개수 (많을수록 안정적, 느림)
  - `maxDepth`: 트리 최대 깊이 (깊을수록 복잡)

**슬라이드 76: Boosting — Gradient Boosted Trees**
- 알고리즘 다이어그램:
  1. 첫 번째 약한 트리 학습
  2. 잔차(residual) 계산 (실제 - 예측)
  3. 잔차를 타겟으로 두 번째 트리 학습
  4. 반복: 이전 오차를 보정하는 새 트리 추가
  5. 최종 예측 = 모든 트리 예측의 가중합
- 주요 하이퍼파라미터:
  - `maxIter`: 반복 횟수 (=트리 개수)
  - `maxDepth`: 각 트리의 최대 깊이
  - `stepSize` (학습률): 각 트리의 기여도

**슬라이드 77: Random Forest vs GBT 비교**
- 비교표:

| 항목 | Random Forest | GBT |
|------|--------------|-----|
| 학습 방식 | 병렬 (독립적 트리) | 순차 (이전 오차 보정) |
| 과적합 경향 | 낮음 (배깅 효과) | 높음 (데이터에 적응) |
| 학습 속도 | 빠름 (병렬화 가능) | 느림 (순차적) |
| 해석가능성 | Feature Importance | Feature Importance |
| 노이즈 민감도 | 낮음 | 높음 |
| 일반적 성능 | 좋음 | 매우 좋음 (튜닝 시) |

**슬라이드 78: Feature Importance (피처 중요도)**
- 개념: 각 피처가 모델 예측에 기여하는 정도 (합계 = 1.0)
- 계산 방법: 불순도 감소(Impurity Decrease) 기반
- 해석 방법:
  - 중요도 높은 피처 = 예측에 큰 영향
  - 중요도 낮은 피처 = 제거 고려 가능
- 코드 예시:
  ```python
  print("RF feature importances:", rf_model.featureImportances)
  ```
- 막대 그래프 시각화 예시

**슬라이드 79: 섹션 10 핵심 정리**
- Decision Tree → 앙상블로 성능 향상
- Bagging (RF): 병렬, 분산 감소, 안정적
- Boosting (GBT): 순차, 편향 감소, 고성능
- Feature Importance: 모델 해석의 핵심 도구

---

### 섹션 11: RF/GBT 분류 실습 (슬라이드 80~84) — 노트북: 080, 090

**슬라이드 80: [실습] 080_RF_GBT_Classification 소개**
- 학습 목표:
  1. RandomForestClassifier, GBTClassifier 학습
  2. 모델 성능 비교 (AUC, Accuracy)
  3. Feature Importance 확인
- 데이터셋: `Social_Network_Ads.csv` (동일)

**슬라이드 81: RandomForestClassifier 코드**
- 코드:
  ```python
  rf = RandomForestClassifier(
      featuresCol="features",
      labelCol="Purchased",
      seed=42
  )
  rf_model = rf.fit(train_data)
  rf_preds = rf_model.transform(test_data)
  ```
- 평가:
  ```python
  rf_auc = auc_eval.evaluate(rf_preds)
  rf_acc = acc_eval.evaluate(rf_preds)
  ```

**슬라이드 82: GBTClassifier 코드**
- 코드:
  ```python
  gbt = GBTClassifier(
      featuresCol="features",
      labelCol="Purchased",
      seed=42
  )
  gbt_model = gbt.fit(train_data)
  gbt_preds = gbt_model.transform(test_data)
  ```

**슬라이드 83: 모델 비교 & Feature Importance**
- 성능 비교표:

| 모델 | AUC | Accuracy |
|------|-----|----------|
| Logistic Regression (Baseline) | 0.89xx | 0.85xx |
| Random Forest | 0.92xx | 0.88xx |
| GBT | 0.93xx | 0.89xx |

- Feature Importance:
  ```python
  print("RF feature importances:", rf_model.featureImportances)
  # 예: (4,[0,1,2],[0.35, 0.55, 0.10])
  # → EstimatedSalary가 가장 중요, 다음 Age
  ```

**슬라이드 84: [실습] 090_Classification_Tuning_Compare — RF/GBT 튜닝 + MLflow**
- 핵심 워크플로우:
  ```
  RF + ParamGrid(numTrees, maxDepth) + CrossValidator + MLflow
  GBT + ParamGrid(maxDepth, maxIter) + CrossValidator + MLflow
  → search_runs()로 최종 비교 → 최고 AUC 모델 선정
  ```
- RF 튜닝 코드:
  ```python
  rf_grid = (ParamGridBuilder()
      .addGrid(rf.numTrees, [20, 50])
      .addGrid(rf.maxDepth, [3, 5])
      .build())
  rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_grid,
                         evaluator=evaluator, numFolds=3, seed=42)

  with mlflow.start_run(run_name="rf_tuned"):
      rf_cv_model = rf_cv.fit(train_data)
      ...
  ```
- GBT 튜닝: `maxDepth=[3,5]`, `maxIter=[20,50]` 그리드
- 최종 선정: "AUC가 가장 높은 모델 = 최종 분류 모델"

---

### 섹션 12: 회귀 분석 이론 (슬라이드 85~89)

**슬라이드 85: 회귀 vs 분류**
- 비교표:

| 구분 | 분류 (Classification) | 회귀 (Regression) |
|------|---------------------|-------------------|
| 출력 | 범주 (0/1, A/B/C) | 연속값 (가격, 온도) |
| 예시 | 구매 예측, 스팸 분류 | 주택 가격, 매출 예측 |
| 평가 지표 | AUC, Accuracy, F1 | RMSE, MAE, R2 |
| 알고리즘 | LR, RF, GBT Classifier | LR, RF, GBT Regressor |

**슬라이드 86: 선형 회귀 (Linear Regression)**
- 수식: y = w1*x1 + w2*x2 + ... + wn*xn + b
- 비용 함수: MSE = (1/n) * sum((y_actual - y_pred)^2)
- 최적화: OLS (최소제곱법) 또는 경사하강법
- 그래프: 2D 산점도 + 회귀선

**슬라이드 87: Feature Scaling의 중요성**
- 문제: Age(0~100) vs Salary(0~200,000) → 스케일 차이가 회귀 계수에 영향
- StandardScaler 효과:
  - 적용 전: 큰 스케일 피처가 지배
  - 적용 후: 모든 피처가 동등한 기여
- 실습에서: `StandardScaler(inputCol="features", outputCol="scaled_features")`

**슬라이드 88: 회귀 평가 지표**
- **RMSE** (Root Mean Squared Error):
  - 수식: sqrt((1/n) * sum((y - y_hat)^2))
  - 해석: 예측 오차의 평균적 크기 (큰 오차에 민감)
  - 낮을수록 좋음
- **MAE** (Mean Absolute Error):
  - 수식: (1/n) * sum(|y - y_hat|)
  - 해석: 직관적 오차 크기 (이상치에 덜 민감)
  - 낮을수록 좋음
- **R2** (결정 계수):
  - 수식: 1 - (SS_res / SS_tot)
  - 해석: 모델이 데이터 분산의 몇 %를 설명하는지
  - 1에 가까울수록 좋음 (0~1)

**슬라이드 89: 회귀에서의 앙상블**
- RandomForestRegressor: 비선형 관계 포착, 과적합 방지
- GBTRegressor: 순차 학습으로 정확도 극대화
- 분류와의 차이:
  - 출력: 확률 → 연속값
  - 평가: AUC → RMSE/MAE/R2
  - 나머지 파이프라인 동일 (VectorAssembler, StandardScaler 등)

---

### 섹션 13: 회귀 실습 (슬라이드 90~95) — 노트북: 100_Regression_LR_RF_GBT

**슬라이드 90: [실습] 100_Regression_LR_RF_GBT 소개**
- 학습 목표:
  1. California Housing 데이터로 회귀 파이프라인 구축
  2. LR / RF / GBT 세 알고리즘 비교
  3. Feature Importance 해석
- 데이터셋: California Housing (sklearn 내장, 20,640건, 8개 피처)
  - 피처: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
  - 타겟: PRICE (중위 주택 가격, 단위: $100,000)

**슬라이드 91: 전처리 파이프라인 (회귀)**
- 코드:
  ```python
  from sklearn.datasets import fetch_california_housing
  import pandas as pd

  # 데이터 로드 (sklearn → Pandas → Spark DataFrame)
  housing = fetch_california_housing()
  pdf = pd.DataFrame(housing.data, columns=housing.feature_names)
  pdf["PRICE"] = housing.target
  spark_df = spark.createDataFrame(pdf)

  # 전처리 (범주형 없음 → VectorAssembler + StandardScaler만)
  feature_cols = [c for c in spark_df.columns if c != "PRICE"]
  assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
  scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
  prep = Pipeline(stages=[assembler, scaler])

  df_ready = prep.fit(spark_df).transform(spark_df)
  data = df_ready.select("scaled_features", "PRICE") \
      .withColumnRenamed("scaled_features", "features")
  ```
- 포인트: 분류와 달리 StringIndexer/OneHotEncoder 불필요 (수치형만)

**슬라이드 92: LinearRegression 베이스라인**
- 코드:
  ```python
  lr = LinearRegression(featuresCol="features", labelCol="PRICE")
  lr_model = lr.fit(train_data)
  lr_preds = lr_model.transform(test_data)

  rmse_eval = RegressionEvaluator(labelCol="PRICE", metricName="rmse")
  mae_eval = RegressionEvaluator(labelCol="PRICE", metricName="mae")
  r2_eval = RegressionEvaluator(labelCol="PRICE", metricName="r2")

  print(f"RMSE: {rmse_eval.evaluate(lr_preds):.4f}")
  print(f"MAE:  {mae_eval.evaluate(lr_preds):.4f}")
  print(f"R2:   {r2_eval.evaluate(lr_preds):.4f}")
  ```

**슬라이드 93: RF / GBT Regressor**
- RandomForestRegressor:
  ```python
  rf = RandomForestRegressor(featuresCol="features", labelCol="PRICE", seed=42)
  rf_model = rf.fit(train_data)
  rf_preds = rf_model.transform(test_data)
  ```
- GBTRegressor:
  ```python
  gbt = GBTRegressor(featuresCol="features", labelCol="PRICE", seed=42)
  gbt_model = gbt.fit(train_data)
  gbt_preds = gbt_model.transform(test_data)
  ```

**슬라이드 94: Feature Importance (회귀)**
- 코드:
  ```python
  print("Random Forest Feature Importances:")
  for feature, importance in zip(feature_cols, rf_model.featureImportances):
      print(f"  {feature:20s}: {importance:.4f}")
  ```
- 예시 해석: MedInc(중위 소득)이 가장 높은 중요도 → 소득이 주택 가격의 핵심 결정 요인

**슬라이드 95: 최종 비교표**
- 코드:
  ```python
  results = {
      "Model": ["Linear Regression", "Random Forest", "GBT"],
      "RMSE": [lr_rmse, rf_rmse, gbt_rmse],
      "MAE": [lr_mae, rf_mae, gbt_mae],
      "R2": [lr_r2, rf_r2, gbt_r2]
  }
  pd.DataFrame(results)
  ```
- 결과 테이블:

| Model | RMSE | MAE | R2 |
|-------|------|-----|-----|
| Linear Regression | 0.74xx | 0.53xx | 0.58xx |
| Random Forest | 0.52xx | 0.34xx | 0.79xx |
| GBT | 0.48xx | 0.32xx | 0.82xx |

- 해석: GBT가 가장 높은 R2, 가장 낮은 RMSE/MAE

---

### 섹션 14: 모델 저장 & 트레이드오프 (슬라이드 96~100) — 노트북: 120_Model_Save_Load_Versioning

**슬라이드 96: [실습] 120_Model_Save_Load_Versioning 소개**
- 학습 목표:
  1. PipelineModel 저장/로드
  2. 버전 관리 전략
  3. 저장된 모델로 추론 실행
- 핵심: "학습된 모델을 저장하지 않으면 매번 재학습 필요"

**슬라이드 97: PipelineModel 저장 / 로드**
- 저장 코드:
  ```python
  # 전처리 + LR을 포함한 완전한 파이프라인
  pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, lr])
  model = pipeline.fit(df)

  # 모델 저장
  path_v1 = os.path.join(MODEL_DIR, "lr_pipeline_v1")
  model.write().overwrite().save(path_v1)
  ```
- 로드 코드:
  ```python
  from pyspark.ml.pipeline import PipelineModel

  loaded = PipelineModel.load(path_v1)
  predictions = loaded.transform(df.limit(10))
  predictions.select("Age", "EstimatedSalary", "prediction").show()
  ```
- 핵심: 전처리 단계까지 포함하여 저장 → 새 데이터에 바로 적용 가능

**슬라이드 98: 버전 관리 전략**
- 디렉토리 구조:
  ```
  saved_models/
  ├── lr_pipeline_v1/    # 1차 모델
  ├── lr_pipeline_v2/    # 2차 모델 (하이퍼파라미터 변경)
  └── rf_pipeline_v1/    # RF 모델
  ```
- 버전 기록표 양식:

| 버전 | 모델 | 파라미터 | AUC | 배포일 | 비고 |
|------|------|---------|-----|--------|------|
| v1 | LR | default | 0.89 | 2024-01-15 | 베이스라인 |
| v2 | LR | regParam=0.01 | 0.91 | 2024-01-16 | 튜닝 후 |

**슬라이드 99: 성능-복잡도-비용 트레이드오프**
- 3축 비교 다이어그램 (또는 비교표):

| 항목 | Linear Regression | Random Forest | GBT |
|------|------------------|---------------|-----|
| 정확도 | 보통 | 좋음 | 매우 좋음 |
| 학습 시간 | 빠름 | 보통 | 느림 |
| 해석 가능성 | 높음 (계수 해석) | 보통 (피처 중요도) | 낮음 |
| 과적합 위험 | 낮음 | 낮음 | 높음 |
| 배포 복잡도 | 낮음 | 보통 | 보통 |
| 추천 상황 | 빠른 프로토타입, 해석 중요 | 범용적, 안정적 | 최고 성능 필요 시 |

**슬라이드 100: 모델 선택 의사결정 프레임워크**
- 의사결정 트리 형태 다이어그램:
  ```
  해석가능성이 중요한가?
  ├── Yes → Linear Regression / Logistic Regression
  └── No → 최고 성능이 필요한가?
       ├── Yes → GBT (+ 충분한 튜닝)
       └── No → Random Forest (안정적, 빠름)
  ```
- 추가 고려사항:
  - 데이터 크기: 소규모 → LR, 대규모 → RF/GBT
  - 배포 환경: 실시간 추론 → 가벼운 모델 선호
  - 규제 요건: 모델 설명 필요 → 해석 가능한 모델

---

### 섹션 15: 팀 미니 프로젝트 & 마무리 (슬라이드 101~105)

**슬라이드 101: 팀 미니 프로젝트 안내**
- 목적: 2일간 배운 End-to-End ML 워크플로우를 직접 적용
- 팀 구성: 2~3인 1팀
- 시간: 약 60~90분 (구현 40분 + 발표 20분 + 피드백)

**슬라이드 102: 프로젝트 가이드라인**
- **데이터셋 옵션** (Data 폴더에서 선택):
  1. `onlinefoods.csv` — 온라인 음식 주문 예측 (분류)
  2. `retail_sales_dataset.csv` — 소매 매출 예측 (회귀)
  3. `titanic.csv` — 생존 예측 (분류)
- **필수 단계:**
  1. 데이터 탐색 (EDA): 스키마 확인, 기술 통계, 분포 확인
  2. 전처리 파이프라인 구축: StringIndexer, VectorAssembler, Scaler
  3. 베이스라인 모델 학습 (LR)
  4. 개선 모델 학습 (RF 또는 GBT) + CrossValidator 튜닝
  5. MLflow로 실험 기록
  6. 최종 비교표 작성 & 모델 선정 근거
- **산출물:**
  - 튜닝 전/후 성능 비교표
  - MLflow 실험 캡처
  - 모델 선정 근거서 (1~2문단)

**슬라이드 103: 2일 과정 전체 정리**
- End-to-End ML 워크플로우 전체 흐름도:
  ```
  데이터 수집 (CSV/Parquet)
      ↓
  데이터 탐색 (DataFrame/SQL/EDA)
      ↓
  전처리 파이프라인 (StringIndexer → OneHot → Assembler → Scaler)
      ↓
  베이스라인 모델 (LR)
      ↓
  교차검증 & 튜닝 (CrossValidator + ParamGrid)
      ↓
  앙상블 모델 (RF / GBT)
      ↓
  실험 추적 (MLflow)
      ↓
  모델 저장 & 버전 관리
      ↓
  모델 선정 (트레이드오프 분석)
  ```
- 학습한 핵심 개념 요약 (키워드 클라우드 형태):
  SparkSession, DataFrame, Parquet, Spark SQL, Pipeline, Transformer, Estimator, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, LogisticRegression, RandomForest, GBT, CrossValidator, ParamGridBuilder, MLflow, RegressionEvaluator, BinaryClassificationEvaluator, Feature Importance, AUC, RMSE, R2

**슬라이드 104: 참고 자료 및 추가 학습**
- PySpark 공식 문서: https://spark.apache.org/docs/latest/api/python/
- Spark MLlib 가이드: https://spark.apache.org/docs/latest/ml-guide.html
- MLflow 공식 문서: https://mlflow.org/docs/latest/
- 추천 후속 학습:
  - 딥러닝 / 시계열 분석 고급 과정 (후수 과정)
  - Spark Structured Streaming
  - Databricks Community Edition 실습

**슬라이드 105: Q&A / 수료**
- 질의응답 시간
- 과정 수료 안내
- 감사 인사

---

## 프롬프트 끝
