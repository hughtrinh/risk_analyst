import sys
import os
import traceback
import argparse

from pyspark.sql import functions as F
from pyspark.sql.types import (
    LongType,
    IntegerType,
    StringType,
    DecimalType
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )


current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

config_path = os.path.abspath(
    os.path.join(current_dir, "..", "..")
)

sys.path.insert(0, config_path)


from _002_utils._00201_utils import (
    create_spark_session,
    read_df_from_parquet,
    write_df_to_parquet
)
from _001_config._00102_path_config import data_path


def clean_money_column(column_name: str):
    """
    Chuẩn hóa trường tiền tệ trước khi ép kiểu Decimal.

    Ví dụ:
    $12,345.67 -> 12345.67
    """

    return (
        F.regexp_replace(
            F.trim(
                F.col(column_name).cast("string")
            ),
            r"[^0-9.\-]",
            ""
        )
        .cast(DecimalType(18, 2))
    )


def _0030203_prc_mdm_users(etl_date: str):
    spark = None
    df = None

    try:
        spark = create_spark_session(
            "_0030203_prc_mdm_users"
        )

        input_path = os.path.join(
            data_path["root_path"],
            "stg",
            "users",
            f"etl_date={etl_date}"
        )

        df = read_df_from_parquet(
            spark=spark,
            parquet_path=input_path
        )

        if df is None:
            print(
                f"Không thể đọc dữ liệu Parquet từ: "
                f"{input_path}"
            )
            return False

        required_columns = {
            "id",
            "current_age",
            "birth_year",
            "birth_month",
            "gender",
            "address",
            "latitude",
            "longitude",
            "per_capita_income",
            "yearly_income",
            "total_debt",
            "credit_score",
            "num_credit_cards"
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            print(
                "Dữ liệu nguồn thiếu các cột bắt buộc: "
                f"{sorted(missing_columns)}"
            )
            return False

        df = df.cache()

        source_count = df.count()

        print("Dữ liệu đọc từ Parquet - stg users:")
        print(
            f"Số lượng bản ghi nguồn: {source_count}"
        )

        df.show(
            5,
            truncate=False
        )

        gender_code = (
            F.when(
                F.upper(
                    F.trim(F.col("gender"))
                ) == "MALE",
                F.lit(1)
            )
            .when(
                F.upper(
                    F.trim(F.col("gender"))
                ) == "FEMALE",
                F.lit(0)
            )
            .otherwise(
                F.lit(None).cast(IntegerType())
            )
        )

        df_mdm_users = (
            df.select(
                F.col("id")
                    .cast(LongType())
                    .alias("user_id"),

                F.col("current_age")
                    .cast(IntegerType())
                    .alias("current_age"),

                F.col("birth_year")
                    .cast(IntegerType())
                    .alias("birth_year"),

                F.col("birth_month")
                    .cast(IntegerType())
                    .alias("birth_month"),

                F.initcap(
                    F.trim(F.col("gender"))
                )
                .cast(StringType())
                .alias("gender_text"),

                gender_code.alias("gender"),

                F.trim(
                    F.col("address")
                )
                .cast(StringType())
                .alias("address"),

                F.col("latitude")
                    .cast(DecimalType(10, 6))
                    .alias("latitude"),

                F.col("longitude")
                    .cast(DecimalType(10, 6))
                    .alias("longitude"),

                clean_money_column(
                    "per_capita_income"
                ).alias("per_capita_income"),

                clean_money_column(
                    "yearly_income"
                ).alias("year_income"),

                F.lit("USD")
                    .cast(StringType())
                    .alias("currency_code"),

                clean_money_column(
                    "total_debt"
                ).alias("total_debt"),

                F.col("credit_score")
                    .cast(IntegerType())
                    .alias("credit_score"),

                F.col("num_credit_cards")
                    .cast(IntegerType())
                    .alias("number_credit_cards"),

                F.current_timestamp()
                    .alias("etl_date")
            )
            .dropDuplicates(["user_id"])
        )

        user_count = df_mdm_users.count()

        print("Dữ liệu sau mapping - mdm_users:")
        print(
            f"Số lượng người dùng sau xử lý: "
            f"{user_count}"
        )

        df_mdm_users.show(
            5,
            truncate=False
        )

        null_user_id_count = (
            df_mdm_users
            .filter(
                F.col("user_id").isNull()
            )
            .count()
        )

        if null_user_id_count > 0:
            print(
                f"Cảnh báo: Có {null_user_id_count} "
                "bản ghi có user_id null."
            )
            return False

        duplicate_user_count = (
            df_mdm_users
            .groupBy("user_id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        if duplicate_user_count > 0:
            print(
                f"Cảnh báo: Có {duplicate_user_count} "
                "user_id bị trùng sau xử lý."
            )
            return False

        output_path = os.path.join(
            data_path["root_path"],
            "prc",
            "mdm_users",
            f"etl_date={etl_date}"
        )

        success_parquet = write_df_to_parquet(
            df=df_mdm_users,
            parquet_path=output_path,
            mode_write="overwrite"
        )

        if not success_parquet:
            print(
                "Lỗi khi ghi dữ liệu "
                "mdm_users vào Parquet."
            )
            return False

        print(
            "Dữ liệu mdm_users đã được ghi "
            f"vào Parquet tại: {output_path}"
        )

        return True

    except Exception as e:
        print(
            "Lỗi trong quá trình ETL "
            f"mdm_users: {str(e)}"
        )

        traceback.print_exc()
        return False

    finally:
        if df is not None:
            try:
                df.unpersist()
            except Exception:
                pass

        if spark is not None:
            spark.stop()
            print("Đã đóng SparkSession.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="_0030203_prc_mdm_users"
    )

    parser.add_argument(
        "--etl_date",
        type=str,
        required=True,
        help="etl_date theo định dạng YYYYMMDD"
    )

    args = parser.parse_args()

    success = _0030203_prc_mdm_users(
        etl_date=args.etl_date
    )

    sys.exit(0 if success else 1)