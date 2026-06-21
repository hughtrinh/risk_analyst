import sys
import os
import traceback
import argparse

from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType

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


def _0030202_prc_mdm_merchant(etl_date: str):
    spark = None
    df = None

    try:
        spark = create_spark_session(
            "_0030202_prc_mdm_merchant"
        )

        input_path = os.path.join(
            data_path["root_path"],
            "stg",
            "transactions",
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
            "merchant_id",
            "merchant_city",
            "merchant_state",
            "zip",
            "use_chip"
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            print(
                "Thiếu các cột bắt buộc trong dữ liệu nguồn: "
                f"{sorted(missing_columns)}"
            )
            return False

        # Cache vì DataFrame được sử dụng bởi nhiều action:
        # count(), show() và write().
        df = df.cache()

        source_count = df.count()

        print("Dữ liệu đọc từ Parquet - stg transactions:")
        print(
            f"Số lượng bản ghi nguồn: {source_count}"
        )

        df.show(5, truncate=False)

        merchant_key = F.pmod(
            F.xxhash64(
                F.concat_ws(
                    "||",
                    F.coalesce(
                        F.col("merchant_id").cast("string"),
                        F.lit("")
                    ),
                    F.coalesce(
                        F.col("merchant_city").cast("string"),
                        F.lit("")
                    ),
                    F.coalesce(
                        F.col("merchant_state").cast("string"),
                        F.lit("")
                    ),
                    F.coalesce(
                        F.col("zip").cast("string"),
                        F.lit("")
                    )
                )
            ),
            F.lit(9223372036854775807)
        ).cast(LongType())

        location_type = (
            F.when(
                (F.col("use_chip") == "Online Transaction")
                | (
                    F.upper(
                        F.trim(F.col("merchant_city"))
                    ) == "ONLINE"
                ),
                F.lit("ONLINE")
            )
            .when(
                F.col("use_chip").isin(
                    "Chip Transaction",
                    "Swipe Transaction"
                )
                & F.col("merchant_city").isNotNull()
                & (
                    F.trim(F.col("merchant_city")) != ""
                )
                & (
                    F.upper(
                        F.trim(F.col("merchant_city"))
                    ) != "ONLINE"
                ),
                F.lit("PHYSICAL")
            )
            .otherwise(F.lit("UNKNOWN"))
        )

        df_mdm_merchant = (
            df.select(
                merchant_key.alias("merchant_key"),

                F.col("merchant_id")
                    .cast(StringType())
                    .alias("merchant_id"),

                F.trim(
                    F.col("merchant_city")
                )
                .cast(StringType())
                .alias("merchant_city"),

                F.upper(
                    F.trim(F.col("merchant_state"))
                )
                .cast(StringType())
                .alias("merchant_state"),

                F.col("zip")
                    .cast(StringType())
                    .alias("zip"),

                location_type.alias("location_type"),

                F.current_timestamp().alias("etl_date")
            )
            .dropDuplicates(["merchant_key"])
        )

        merchant_count = df_mdm_merchant.count()

        print("Dữ liệu sau mapping - mdm_merchant:")
        print(
            f"Số lượng merchant duy nhất: "
            f"{merchant_count}"
        )

        df_mdm_merchant.show(
            5,
            truncate=False
        )

        null_key_count = (
            df_mdm_merchant
            .filter(F.col("merchant_key").isNull())
            .count()
        )

        if null_key_count > 0:
            print(
                "Cảnh báo: Có "
                f"{null_key_count} bản ghi merchant_key null."
            )
            return False

        output_path = os.path.join(
            data_path["root_path"],
            "prc",
            "mdm_merchant",
            f"etl_date={etl_date}"
        )

        success_parquet = write_df_to_parquet(
            df=df_mdm_merchant,
            parquet_path=output_path,
            mode_write="overwrite"
        )

        if not success_parquet:
            print(
                "Lỗi khi ghi dữ liệu "
                "mdm_merchant vào Parquet."
            )
            return False

        print(
            "Dữ liệu mdm_merchant đã được ghi "
            f"vào Parquet tại: {output_path}"
        )

        return True

    except Exception as e:
        print(
            "Lỗi trong quá trình ETL "
            f"mdm_merchant: {str(e)}"
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
        description="_0030202_prc_mdm_merchant"
    )

    parser.add_argument(
        "--etl_date",
        type=str,
        required=True,
        help="etl_date theo định dạng YYYYMMDD"
    )

    args = parser.parse_args()

    success = _0030202_prc_mdm_merchant(
        etl_date=args.etl_date
    )

    sys.exit(0 if success else 1)