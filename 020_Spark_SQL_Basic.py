# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="pSxugzbzT2Kl"
# # Spark SQL
# - Spark SQL 구문 및 패턴은 다른 최신 데이터베이스 시스템에서 사용하는 SQL과 동일

# %% id="Z-JCFmohTnz7"
import os
import sys
from pyspark.sql import SparkSession

# Colab 환경인지 확인
IN_COLAB = "google.colab" in sys.modules
# Colab이면 /content, 아니면 현재 작업 디렉토리를 BASE로 설정
BASE = "/content" if IN_COLAB else os.getcwd()

# SparkSession 생성 (PySpark 애플리케이션의 진입점)
# 데이터프레임 생성, 데이터 읽기/쓰기, SQL 작업 등을 수행
spark = SparkSession.builder.appName("Spark_SQL_Basic").getOrCreate()

# %% [markdown] id="2la1GkOiUYka"
# ## 테이블 만들기
#
# 이 과정에서 다양한 파일(및 파일 형식)로 작업할 것입니다. SQL 인터페이스를 통해 데이터에 액세스하기 위해 가장 먼저 해야 할 일은 해당 데이터에서 **테이블**을 생성하는 것입니다.
#
#  Spark SQL을 사용하여 테이블을 쿼리할 것입니다. 이 테이블에는 이름과 성, 생년월일, 급여 등과 같은 사람에 대한 사실을 담고 있는 가상 레코드가 포함되어 있습니다. 우리는 많은 빅 데이터 워크로드에서 일반적으로 사용된는 `Parquet` 파일 형식을 사용합니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="25tSaE0ZUZv_" outputId="07ef9611-17a5-4757-9e15-f9ae5619a6d2"
# Google Drive에서 파일 경로 지정
file_path = "/content/people_10M1.parquet"

# SQL 명령 실행
# 1. 기존 테이블 삭제
spark.sql("DROP TABLE IF EXISTS People10M")

# 2. 새로운 테이블 생성 및 Parquet 데이터 로드
create_table_query = f"""
CREATE TABLE People10M
USING parquet
OPTIONS (
  path '{file_path}',
  header 'false'
)
"""
spark.sql(create_table_query)

# 테이블 내용 확인
spark.sql("SELECT * FROM People10M LIMIT 5").show()

# %% colab={"base_uri": "https://localhost:8080/"} id="c_omlLqIqJp5" outputId="798c21aa-ba39-4ba5-f178-5d1a8ccb1015"
# People10M 테이블의 전체 행(row) 수를 계산
spark.sql("SELECT count(*) FROM People10M").show()

# %% colab={"base_uri": "https://localhost:8080/"} id="ixP-QiRRs0xy" outputId="31e64efc-396a-421f-c075-545750ed097b"
# People10M 테이블에서 lastName 컬럼의 데이터를 상위 5개 행만 조회
spark.sql("SELECT lastName FROM People10M LIMIT 5").show()

# %% [markdown] id="q5_7EgPVtOrR"
# `DESCRIBE` 명령을 사용하여 이 테이블의 스키마를 볼 수 있습니다.
#
# **스키마**는 테이블의 열과 해당 열 내의 데이터 유형을 정의하는 목록입니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="s_8x3vKztKYJ" outputId="22fa7adf-1105-417b-821f-7cfa78e09241"
# People10M 테이블의 스키마 정보를 조회
spark.sql("DESCRIBE People10M").show()

# %% [markdown] id="pRM4kJ_JtjJq"
# ## 쿼리 결과 표시
#
# `SELECT` 문으로 시작하는 쿼리는 자동으로 아래 결과를 표시합니다. 'WHERE' 절을 사용하여 주어진 조건이나 조건 집합을 충족하는 결과로 결과를 제한할 수 있습니다.
#
# 다음 쿼리의 경우 결과 열을 `firstName`, `middleName`, `lastName` 및 `birthdate`로 제한합니다. 마지막에 'WHERE' 절을 사용하여 성별이 'F'로 나열된 1990년 이후에 태어난 사람들로 결과 집합을 제한 합니다.
#
# `birthDate`는 타임스탬프 유형이므로 `YEAR()` 함수를 사용하여 생년월일을 추출할 수 있습니다.

# %% colab={"base_uri": "https://localhost:8080/", "height": 423} id="tBB7SxVGtfaZ" outputId="08a659dd-a1fb-4a2f-ce6d-65b76cd29a33"
# People10M 테이블에서 gender가 'F'(여성)인 데이터를 조회
spark.sql(
    """
    SELECT firstName, middleName, lastName, birthdate, gender
    FROM People10M
    WHERE gender == 'F'
    """
).toPandas()

# %% [markdown] id="7I0O2jOAuX8Z"
# ## 수치 계산
#
# Spark SQL에는 표준 SQL에서도 사용되는 많은 기본 제공 함수가 포함되어 있습니다. 이를 사용하여 규칙에 따라 새 열을 만들 수 있습니다. 여기서는 간단한 수학 함수를 사용하여 salary의 20%를 계산합니다. 키워드 'AS'를 사용하여 새 열을 'savings'로 이름을 바꿉니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="EASOhrO2uPJI" outputId="815418c2-a8f0-4eae-8148-3416c1cdb1ef"
# People10M 테이블에서 데이터를 조회하며, 급여(salary)의 20%를 계산한 새로운 컬럼(savings)을 추가
result = spark.sql("""
SELECT
    id,                 -- 개인 식별 ID
    firstName,          -- 이름
    middleName,         -- 중간 이름
    lastName,           -- 성
    gender,             -- 성별
    birthDate,          -- 생년월일
    ssn,                -- 사회보장번호
    salary,             -- 급여
    (salary * 0.2) AS savings -- 급여의 20%를 계산한 새로운 컬럼 (savings)
FROM People10M
""")

# 상위 5개 행만 결과 출력
result.limit(5).show()

# %% [markdown] id="TjrOeSpTvNRE"
# ## Temporary View 생성
#
# **temporary view**는 데이터 탐색에 유용합니다. 아래 셀에서 마지막 쿼리의 모든 정보를 포함하는 temporary view를 만들고 또 다른 새 column인 `birthYear`를 추가합니다.

# %% id="RPQ5UoIZu7SY"
# People10M 테이블에서 데이터를 조회하며, birthDate 컬럼에서 연도를 추출하여 birthYear라는 새로운 컬럼을 추가하고,
# 결과를 PeopleSavings라는 임시 뷰(Temporary View)로 생성 또는 대체
_ = spark.sql("""
CREATE OR REPLACE TEMPORARY VIEW PeopleSavings AS
SELECT
    id,                 -- 개인 식별 ID
    firstName,          -- 이름
    middleName,         -- 중간 이름
    lastName,           -- 성
    gender,             -- 성별
    birthDate,          -- 생년월일
    ssn,                -- 사회보장번호
    salary,             -- 급여
    YEAR(birthDate) AS birthYear -- 생년월일에서 연도를 추출하여 birthYear 컬럼 추가
FROM People10M
""")

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="uiSiif3Jv3di" outputId="9e157787-6381-4f6b-8bf7-01810cb7f4fd"
spark.sql("SELECT * FROM PeopleSavings limit 5;").toPandas()

# %% [markdown] id="5QTKe2pTxhAH"
# ## Query Views
#
# 대부분의 경우 테이블을 쿼리하는 것과 똑같이 뷰를 쿼리할 수 있습니다. 아래 쿼리는 내장 함수 `AVG()`를 사용하여 `birthYear`로 **grouped by** 된 `avgSalary`를 계산합니다. 이것은 집계 함수로, 일련의 값에 대해 계산을 수행한다는 의미입니다. 요약하려는 값의 하위 집합을 식별하려면 'GROUP BY' 절을 포함해야 합니다.
#
# 마지막 절인 `ORDER BY`는 행이 나타나는 순서를 제어할 열을 선언하고 키워드 `DESC`는 행이 내림차순으로 나타남을 의미합니다.
#
# `AVG()` 주위에 `ROUND()` 함수를 사용하여 가장 가까운 센트로 반올림합니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="PvMsumzWxYLL" outputId="68f5f6e0-1b86-4691-c890-48d409550e11"
# PeopleSavings 임시 뷰에서 birthYear(출생 연도)별로 평균 급여(salary)를 계산하여
# 평균 급여가 높은 순서로 정렬하는 쿼리
result = spark.sql(
    """
    SELECT
        birthYear,                    -- 출생 연도
        ROUND(AVG(salary), 2) AS avgSalary   -- 평균 급여를 소수점 둘째 자리로 반올림하여 avgSalary로 이름 지정
    FROM PeopleSavings               -- 데이터를 조회할 임시 뷰
    GROUP BY birthYear               -- 출생 연도를 기준으로 그룹화
    ORDER BY avgSalary DESC;         -- 평균 급여(avgSalary)가 높은 순서로 정렬
    """
)

result.limit(5)

# %% [markdown] id="yuLvX80WyH3P"
# ## 새 테이블 정의
#
# 이제 Parquet을 사용하여 테이블을 만드는 방법을 보여드리겠습니다. Parquet는 오픈 소스 열 기반(column-based) 파일 형식입니다. Apache Spark는 다양한 파일 형식을 지원합니다. `USING` 키워드로 테이블을 작성하는 방법을 지정할 수 있습니다.
#
#
# 지금은 새 테이블을 만드는 데 사용할 명령에 중점을 둘 것입니다.
#
# 이 데이터에는 1880년부터 2016년까지 연도별로 미국에서 이름의 상대적인 인기도에 대한 정보가 포함되어 있습니다.
#
# `Line 1`: 테이블은 고유한 이름을 가져야 합니다. `DROP TABLE IF EXISTS` 명령을 포함하면 이 테이블이 이미 생성된 경우에도 다음 행(`CREATE TABLE`)이 성공적으로 실행될 수 있습니다. 줄 끝에 있는 세미콜론을 사용하면 동일한 셀에서 다른 명령을 실행할 수 있습니다.
#
# `Line 2`: `ssaNames`라는 테이블을 생성하고 데이터 소스 유형(`parquet`)을 정의하며 따라야 할 몇 가지 선택적 매개변수가 있음을 나타냅니다..
#
# `Line 3`: 오브젝트 스토리지의 파일 경로를 식별합니다.
#
# `Line 4`: 테이블의 첫 번째 줄이 헤더로 처리되어야 함을 나타냅니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="OzztcotR2W9o" outputId="3c6bc4f2-3f53-465f-d25a-f7a01d96801e"
# 기존에 ssaNames라는 테이블이 존재하면 삭제
spark.sql("""
    DROP TABLE IF EXISTS ssaNames
""")

# ssaNames라는 새로운 테이블 생성
# 테이블은 csv 파일을 데이터 소스로 사용하며, 해당 테이블의 메타데이터는 스파크의 카탈로그에 저장됩니다.
spark.sql("""
    CREATE TABLE ssaNames
    USING csv                -- csv 파일 형식을 사용
    OPTIONS (                -- csv 파일 관련 옵션 지정
        path '/content/babyNamesUSYOB-mostpopular.csv',
        header 'true'        -- 첫 번째 행을 컬럼 헤더로 사용
    )
""")

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="0R9SDE4K0zZY" outputId="fe068274-251c-4657-e4b2-2ef2bd0b458a"
spark.sql("SELECT * FROM ssaNames LIMIT 5").toPandas()

# %% [markdown] id="o7bp7Kxh3sKU"
# ## 두 테이블 join
#
# 테이들블을 결합하여 데이터가 어떻게 관련되어 있는지 파악할 수 있습니다. 예를 들어, 다음과 같은 내용이 궁금할 수 있습니다.
# > 이름과 성, 생년월일, 급여 등이 저장된 `People10M` 데이터 세트에 얼마나 많은 흔한 이름이 나타납니까?
#
# join을 사용하여 이 질문에 답할 것입니다. 다음 일련의 단계로 join을 수행합니다.
# - 인기 있는 이름이 저장된 `ssaNames` table에서 중복이 제거된 이름으로 구성된 temporary view 생성
# - `People10M` table에서 중목이 제거된 이름으로 구성된 temporary view 생성  
# - 두개의 temporary view에 대해 join 실행

# %% colab={"base_uri": "https://localhost:8080/"} id="ux7_5KS14q0W" outputId="39b9c860-eabd-4c16-e67d-56372f5e43d6"
# ssaNames 테이블에서 중복이 제거된 이름(Name)으로 구성된 임시 뷰(Temporary View) 생성
# 결과는 UniqueNames라는 임시 뷰에 저장됩니다.
spark.sql("""
CREATE OR REPLACE TEMP VIEW UniqueNames AS
    SELECT DISTINCT Name   -- Name 컬럼에서 중복을 제거한 고유 값만 선택
    FROM ssaNames                -- 데이터를 조회할 원본 테이블
""")


# %% colab={"base_uri": "https://localhost:8080/"} id="oqczY4N-5iKi" outputId="5cb83294-8472-4e27-d9f7-807af4b9f16d"
# People10M 테이블에서 중복이 제거된 이름(firstName)으로 구성된 임시 뷰(Temporary View) 생성
# 컬럼 이름을 Name으로 변경하여 고유한 값만 선택합니다.
# 결과는 UniquePeople10M라는 임시 뷰에 저장됩니다.
spark.sql("""
CREATE OR REPLACE TEMP VIEW UniquePeople10M AS
    SELECT DISTINCT firstName AS Name    -- firstName 컬럼에서 중복을 제거하고 컬럼 이름을 Name으로 변경
    FROM People10M                     -- 데이터를 조회할 원본 테이블
""")

# %% colab={"base_uri": "https://localhost:8080/"} id="7VDVcK9f3oyK" outputId="47f803bd-f887-463f-94dd-ef21c430367d"
# 두 개의 임시 뷰를 조인하여 공통된 이름의 수를 계산하고
# 결과를 common_name_count라는 컬럼 이름으로 반환합니다.
# "JOIN"을 사용하여 UniquePeople10M과 UniqueNames 임시 뷰를 Name 컬럼을 기준으로 조인합니다.
result = spark.sql("""
SELECT COUNT(*) AS common_name_count     -- 조인된 결과에서 공통된 이름의 수를 계산
FROM UniquePeople10M p               -- UniquePeople10M 임시 뷰
JOIN UniqueNames s                         -- UniqueNames 임시 뷰
ON p.Name = s.Name                        -- 두 뷰에서 Name 컬럼을 기준으로 조인
""")

# 결과 출력
# 공통된 이름의 수를 출력
result.show()

# %% [markdown] id="1_Bqtwho7sNb"
# ## temporary view 생성
#
# 다음으로 실제 join을 쉽게 읽고 쓸 수 있도록 두 개의 temporary view 를 생성합니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="sbcbPQQ337Lt" outputId="f1eb2c5f-d7e2-4040-ced1-375772555bb9"
# ssaNames 테이블에서 Name 컬럼의 고유한 값(DISTINCT)을 선택하여 UniqueNames라는 임시 뷰 생성
spark.sql("""
CREATE OR REPLACE TEMP VIEW UniqueNames AS
    SELECT DISTINCT Name       -- Name 컬럼의 중복을 제거한 고유 값만 선택
    FROM ssaNames              -- 데이터를 가져올 원본 테이블
""")

# People10M 테이블에서 firstName 컬럼의 고유한 값(DISTINCT)을 선택하여 PeopleDistinctNames라는 임시 뷰 생성
spark.sql("""
CREATE OR REPLACE TEMPORARY VIEW PeopleDistinctNames AS
  SELECT DISTINCT firstName   -- firstName 컬럼의 중복을 제거한 고유 값만 선택
  FROM People10M              -- 데이터를 가져올 원본 테이블
""")

# %% [markdown] id="n-FbNERh8wDy"
# ## join 실행
#
# 이제 temporary view를 사용하여 두 데이터 세트를 **join**할 수 있습니다.
#
# default로 여기에 표시된 조인 유형은 'INNER'입니다. 즉, 결과에는 두 세트의 교집합이 포함되며 두 세트에 없는 이름은 표시되지 않습니다. 기본값이므로 join 유형을 지정하지 않았습니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="AQP38dGO8wuw" outputId="24d6b792-4502-48bf-c3aa-125ce2204aae"
# PeopleDistinctNames 임시 뷰와 UniqueNames 임시 뷰를 조인하여 공통된 이름(firstName)을 조회하는 쿼리
spark.sql("""
SELECT firstName                      -- 결과로 반환할 컬럼: PeopleDistinctNames 뷰의 firstName 컬럼
FROM PeopleDistinctNames              -- 첫 번째 데이터 소스: PeopleDistinctNames 임시 뷰
JOIN UniqueNames ON firstName = Name  -- 두 번째 데이터 소스: UniqueNames 임시 뷰
                                       -- 조인 조건: PeopleDistinctNames의 firstName과 UniqueNames의 Name 컬럼이 같은 경우
""").show()

# %% [markdown] id="9gHfhCwp9LD5"
# ## How many names?
#
# 이 질문에 답하기 위해 이 조인을 수행하고 결과에 레코드 수를 포함할 수 있습니다.

# %% colab={"base_uri": "https://localhost:8080/"} id="i0WhK2Bb9C6r" outputId="5e774df1-a375-497d-ed2e-b8d98de7b52c"
# PeopleDistinctNames 임시 뷰와 UniqueNames 임시 뷰를 조인하여 공통된 이름의 수를 계산
spark.sql("""
SELECT count(*)                     -- 조인 결과의 전체 행 수(공통된 이름의 개수)를 계산
FROM PeopleDistinctNames            -- 첫 번째 데이터 소스: PeopleDistinctNames 임시 뷰
JOIN UniqueNames ON firstName = Name     -- 두 번째 데이터 소스: UniqueNames 임시 뷰
                                      -- 조인 조건: PeopleDistinctNames의 firstName과 UniqueNames의 Name 컬럼이 동일한 경우
""").show()

# %% id="ae73b1ca"
spark.stop()

# %% id="EMzLFBQf9Uux"
