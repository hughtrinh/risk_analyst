"""
Sample: Kết nối SQL Server bằng PySpark với JDBC
"""
import sys
import os
from pathlib import Path
from typing import Any, Mapping

from pyspark.sql import SparkSession

current_dir = os.path.dirname(__file__)
config_path = os.path.join(current_dir, "..")
config_path = os.path.abspath(config_path)
sys.path.insert(0, config_path)


from _001_config._00101_database_config import *

def get_sqlserver_connection(spark):
    """
    Tạo JDBC DataFrameReader kết nối tới SQL Server.
    """

    config = JDBC_SQLSERVER_URL
    properties = config["properties"]

    reader = (
        spark.read
        .format("jdbc")
        .option("url", config["url"])
        .option("user", properties["user"])
        .option("password", properties["password"])
        .option("driver", properties["driver"])
    )

    return reader

def create_spark_session(
    app_name: str = "risk_analyst",
    master: str | None = "local[*]",
    spark_configs: Mapping[str, Any] | None = None,
    log_level: str = "WARN",
) -> SparkSession:
    """
    Tạo hoặc lấy một SparkSession dùng chung cho project.

    Các JDBC JAR trong ``ORL_LIB_PATH["driver_class_path"]`` sẽ được tự động
    thêm vào driver, executor và ``spark.jars``.

    Args:
        app_name: Tên ứng dụng hiển thị trên Spark UI.
        master: Spark master URL. Dùng ``None`` khi chạy bằng ``spark-submit``
            hoặc trên cluster đã cấu hình master từ bên ngoài.
        spark_configs: Các cấu hình Spark bổ sung, ví dụ
            ``{"spark.sql.shuffle.partitions": "8"}``.
        log_level: Mức log của SparkContext.

    Returns:
        SparkSession đã được khởi tạo.

    Raises:
        FileNotFoundError: Khi JDBC JAR đã cấu hình nhưng không tồn tại.
        RuntimeError: Khi SparkSession không thể khởi tạo.
    """
    builder = SparkSession.builder.appName(app_name)

    if master:
        builder = builder.master(master)

    driver_class_path = ORL_LIB_PATH.get("driver_class_path", "")
    if driver_class_path:
        jar_paths = [
            os.path.abspath(os.path.expandvars(path.strip()))
            for path in driver_class_path.split(os.pathsep)
            if path.strip()
        ]
        missing_jars = [path for path in jar_paths if not os.path.isfile(path)]
        if missing_jars:
            raise FileNotFoundError(
                "Không tìm thấy JDBC JAR: " + ", ".join(missing_jars)
            )

        class_path = os.pathsep.join(jar_paths)
        spark_jars = ",".join(Path(path).as_uri() for path in jar_paths)
        builder = (
            builder
            .config("spark.driver.extraClassPath", class_path)
            .config("spark.executor.extraClassPath", class_path)
            .config("spark.jars", spark_jars)
        )

    for key, value in (spark_configs or {}).items():
        builder = builder.config(key, value)

    try:
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel(log_level)
        return spark
    except Exception as exc:
        raise RuntimeError(
            f"Không thể khởi tạo SparkSession '{app_name}': {exc}"
        ) from exc

def read_df_from_table(spark, table_name=None, script_sql=None, schema=None):
    try:
        if script_sql:
            # Xử lý đọc theo script SQL
            reader = get_sqlserver_connection(spark)
            if schema is not None:
                df = reader.option("query", script_sql).schema(schema).load()
            else:
                df = reader.option("query", script_sql).load()
        else:
            # Xử lý đọc toàn bộ bảng
            reader = get_sqlserver_connection(spark)
            if schema is not None:
                df = reader.option("dbtable", table_name).schema(schema).load()
            else:
                df = reader.option("dbtable", table_name).load()
        return df
    except Exception as e:
        print(f"Error reading table: {str(e)}")
        return None
    
def read_df_from_parquet(spark, parquet_path, schema=None):
    try:
        # spark = create_spark_session()
        if schema is not None:
            df = spark.read.schema(schema).parquet(parquet_path)
        else:
            df = spark.read.parquet(parquet_path)
        return df
    except Exception as e:
        print(f"Error reading parquet file: {str(e)}")
        return None

def read_df_from_csv(spark, csv_path, schema=None):
    try:
        # spark = create_spark_session()
        if schema is not None:
            df = spark.read.csv(csv_path, header=True, schema=schema)
        else:
            df = spark.read.csv(csv_path, header=True, inferSchema=True)
        return df
    except Exception as e:
        print(f"Error read csv file: {str(e)}")
        return None

def write_df_to_parquet(df, parquet_path, mode_write):
    try:
        if str(mode_write).lower() == "overwrite":
            return write_df_to_parquet(df, parquet_path)
        df.write.mode(mode_write).parquet(parquet_path)
        return True
    except Exception as e:
        print(f"Error write parquet file: {str(e)}")
        return False
    
def write_df_to_table(df, table_name, mode_write):
    try:
        df.write.mode(mode_write).jdbc(
            url=JDBC_SQLSERVER_URL["url"], table=table_name, properties=JDBC_SQLSERVER_URL["properties"]
        )
        return True
    except Exception as e:
        print(f"Error write to table: {str(e)}")
        return False
    
def write_df_to_csv(df, csv_path, mode_write):
    try:
        df.write.mode(mode_write).csv(csv_path)
        return True
    except Exception as e:
        print(f"Error write csv file: {str(e)}")
        return False
