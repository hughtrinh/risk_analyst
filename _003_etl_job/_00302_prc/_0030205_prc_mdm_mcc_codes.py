import sys
import os
import traceback
import argparse
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql.types import StringType


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


def validate_etl_date(etl_date: str) -> bool:
    """
    Kiểm tra etl_date có đúng định dạng YYYYMMDD hay không.
    """

    try:
        datetime.strptime(etl_date, "%Y%m%d")
        return True
    except ValueError:
        return False


def _0030205_prc_mdm_mcc_codes(etl_date: str):
    spark = None
    df = None
    df_mdm_mcc_codes = None

    try:
        if not validate_etl_date(etl_date):
            print(
                "etl_date không hợp lệ. "
                "Định dạng yêu cầu: YYYYMMDD"
            )
            return False

        spark = create_spark_session(
            "_0030205_prc_mdm_mcc_codes"
        )

        input_path = os.path.join(
            data_path["root_path"],
            "stg",
            "mcc_codes",
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
            "mcc_id",
            "description"
        }

        missing_columns = (
            required_columns - set(df.columns)
        )

        if missing_columns:
            print(
                "Dữ liệu nguồn thiếu các cột bắt buộc: "
                f"{sorted(missing_columns)}"
            )
            return False

        df = df.cache()

        source_count = df.count()

        print("Dữ liệu đọc từ Parquet - stg mcc_codes:")
        print(
            f"Số lượng bản ghi nguồn: {source_count}"
        )

        df.show(
            5,
            truncate=False
        )

        # Chuẩn hóa MCC thành chuỗi 4 ký tự.
        # Ví dụ: 742 -> 0742.
        normalized_mcc_id = (
            F.when(
                F.col("mcc_id").isNull()
                | (
                    F.trim(
                        F.col("mcc_id").cast("string")
                    ) == ""
                ),
                F.lit(None).cast(StringType())
            )
            .otherwise(
                F.lpad(
                    F.trim(
                        F.col("mcc_id").cast("string")
                    ),
                    4,
                    "0"
                )
            )
        )

        df_prepared = (
            df
            .withColumn(
                "_mcc_id_normalized",
                normalized_mcc_id
            )
            .withColumn(
                "_description_normalized",
                F.trim(
                    F.col("description").cast("string")
                )
            )
        )

        invalid_mcc_count = (
            df_prepared
            .filter(
                F.col("_mcc_id_normalized").isNull()
                | (
                    ~F.col("_mcc_id_normalized")
                    .rlike(r"^\d{4}$")
                )
            )
            .count()
        )

        if invalid_mcc_count > 0:
            print(
                f"Lỗi dữ liệu: Có {invalid_mcc_count} "
                "bản ghi có mcc_id không hợp lệ."
            )

            print("Một số MCC không hợp lệ:")

            (
                df_prepared
                .filter(
                    F.col("_mcc_id_normalized").isNull()
                    | (
                        ~F.col("_mcc_id_normalized")
                        .rlike(r"^\d{4}$")
                    )
                )
                .select(
                    "mcc_id",
                    "_mcc_id_normalized",
                    "description"
                )
                .show(
                    20,
                    truncate=False
                )
            )

            return False

        long_description_count = (
            df_prepared
            .filter(
                F.length(
                    F.col("_description_normalized")
                ) > 1000
            )
            .count()
        )

        if long_description_count > 0:
            print(
                f"Cảnh báo: Có {long_description_count} "
                "bản ghi description dài quá 1000 ký tự. "
                "Dữ liệu sẽ được cắt còn 1000 ký tự."
            )

        conflicting_mcc_count = (
            df_prepared
            .groupBy("_mcc_id_normalized")
            .agg(
                F.countDistinct(
                    "_description_normalized"
                ).alias("description_count")
            )
            .filter(
                F.col("description_count") > 1
            )
            .count()
        )

        if conflicting_mcc_count > 0:
            print(
                f"Lỗi dữ liệu: Có {conflicting_mcc_count} "
                "mcc_id có nhiều description khác nhau."
            )

            (
                df_prepared
                .groupBy("_mcc_id_normalized")
                .agg(
                    F.collect_set(
                        "_description_normalized"
                    ).alias("descriptions")
                )
                .filter(
                    F.size(
                        F.col("descriptions")
                    ) > 1
                )
                .show(
                    20,
                    truncate=False
                )
            )

            return False

        df_mdm_mcc_codes = (
            df_prepared
            .select(
                F.col("_mcc_id_normalized")
                .cast(StringType())
                .alias("mcc_id"),

                F.substring(
                    F.col("_description_normalized"),
                    1,
                    1000
                )
                .cast(StringType())
                .alias("description"),

                F.current_timestamp()
                .alias("etl_date")
            )
            .dropDuplicates(["mcc_id"])
            .cache()
        )

        target_count = df_mdm_mcc_codes.count()

        print("Dữ liệu sau mapping - mdm_mcc_codes:")
        print(
            f"Số lượng MCC sau xử lý: {target_count}"
        )

        df_mdm_mcc_codes.show(
            20,
            truncate=False
        )

        null_description_count = (
            df_mdm_mcc_codes
            .filter(
                F.col("description").isNull()
                | (
                    F.trim(
                        F.col("description")
                    ) == ""
                )
            )
            .count()
        )

        if null_description_count > 0:
            print(
                f"Cảnh báo: Có {null_description_count} "
                "bản ghi không có description."
            )

        output_path = os.path.join(
            data_path["root_path"],
            "prc",
            "mdm_mcc_codes",
            f"etl_date={etl_date}"
        )

        print(
            f"Đường dẫn đầu ra: {output_path}"
        )

        success_parquet = write_df_to_parquet(
            df=df_mdm_mcc_codes,
            parquet_path=output_path,
            mode_write="overwrite"
        )

        if not success_parquet:
            print(
                "Lỗi khi ghi dữ liệu "
                "mdm_mcc_codes vào Parquet."
            )
            return False

        print(
            "Dữ liệu mdm_mcc_codes đã được ghi "
            f"vào Parquet tại: {output_path}"
        )

        return True

    except Exception as e:
        print(
            "Lỗi trong quá trình ETL "
            f"mdm_mcc_codes: {str(e)}"
        )

        traceback.print_exc()
        return False

    finally:
        if df_mdm_mcc_codes is not None:
            try:
                df_mdm_mcc_codes.unpersist()
            except Exception:
                pass

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
        description="_0030205_prc_mdm_mcc_codes"
    )

    parser.add_argument(
        "--etl_date",
        type=str,
        required=True,
        help="etl_date theo định dạng YYYYMMDD"
    )

    args = parser.parse_args()

    success = _0030205_prc_mdm_mcc_codes(
        etl_date=args.etl_date
    )

    sys.exit(0 if success else 1)