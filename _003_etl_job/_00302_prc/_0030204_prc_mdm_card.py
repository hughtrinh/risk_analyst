import sys
import os
import traceback
import argparse
from datetime import datetime

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
    Chuẩn hóa dữ liệu tiền tệ trước khi cast Decimal.

    Ví dụ:
    $12,345.67 -> 12345.67
    """

    value = F.trim(
        F.col(column_name).cast("string")
    )

    cleaned_value = F.regexp_replace(
        value,
        r"[^0-9.\-]",
        ""
    )

    return (
        F.when(
            cleaned_value == "",
            F.lit(None)
        )
        .otherwise(cleaned_value)
        .cast(DecimalType(18, 2))
    )


def parse_date_column(
    column_name: str,
    end_of_month: bool = False
):
    """
    Chuyển đổi các định dạng ngày:

    MM/yyyy
    MM/yy
    yyyy-MM
    yyyy-MM-dd
    MM/dd/yyyy
    dd/MM/yyyy

    expires:
        Trả về ngày cuối cùng của tháng.

    acct_open_date:
        Trả về ngày đầu tiên của tháng nếu nguồn chỉ có tháng/năm.
    """

    value = F.trim(
        F.col(column_name).cast("string")
    )

    parsed_date = (
        F.when(
            value.isNull() | (value == ""),
            F.lit(None).cast("date")
        )

        # Ví dụ: 09/2025
        .when(
            value.rlike(r"^\d{2}/\d{4}$"),
            F.to_date(
                F.concat(
                    F.lit("01/"),
                    value
                ),
                "dd/MM/yyyy"
            )
        )

        # Ví dụ: 09/25
        .when(
            value.rlike(r"^\d{2}/\d{2}$"),
            F.to_date(
                F.concat(
                    F.lit("01/"),
                    value
                ),
                "dd/MM/yy"
            )
        )

        # Ví dụ: 2025-09
        .when(
            value.rlike(r"^\d{4}-\d{2}$"),
            F.to_date(
                F.concat(
                    value,
                    F.lit("-01")
                ),
                "yyyy-MM-dd"
            )
        )

        # Các định dạng ngày đầy đủ
        .otherwise(
            F.coalesce(
                F.to_date(value, "yyyy-MM-dd"),
                F.to_date(value, "MM/dd/yyyy"),
                F.to_date(value, "dd/MM/yyyy")
            )
        )
    )

    if end_of_month:
        return F.last_day(parsed_date)

    return parsed_date


def validate_etl_date(etl_date: str):
    """
    Kiểm tra etl_date có đúng định dạng YYYYMMDD hay không.
    """

    try:
        datetime.strptime(etl_date, "%Y%m%d")
        return True

    except ValueError:
        return False


def _0030204_prc_mdm_card(etl_date: str):
    spark = None
    df = None
    df_mdm_cards = None

    try:
        if not validate_etl_date(etl_date):
            print(
                "etl_date không hợp lệ. "
                "Định dạng yêu cầu: YYYYMMDD"
            )
            return False

        spark = create_spark_session(
            "_0030204_prc_mdm_card"
        )

        input_path = os.path.join(
            data_path["root_path"],
            "stg",
            "cards",
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
            "client_id",
            "card_type",
            "card_brand",
            "card_number",
            "expires",
            "cvv",
            "has_chip",
            "num_cards_issued",
            "credit_limit",
            "acct_open_date",
            "year_pin_last_changed"
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

        print("Dữ liệu đọc từ Parquet - stg cards:")
        print(
            f"Số lượng bản ghi nguồn: {source_count}"
        )

        df.show(
            5,
            truncate=False
        )

        duplicate_card_id_count = (
            df
            .filter(F.col("id").isNotNull())
            .groupBy("id")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )

        if duplicate_card_id_count > 0:
            print(
                "Cảnh báo: Có "
                f"{duplicate_card_id_count} card_id "
                "bị trùng trong dữ liệu nguồn."
            )

        has_chip_code = (
            F.when(
                F.upper(
                    F.trim(
                        F.col("has_chip").cast("string")
                    )
                ).isin(
                    "YES",
                    "Y",
                    "TRUE",
                    "1"
                ),
                F.lit(1)
            )
            .when(
                F.upper(
                    F.trim(
                        F.col("has_chip").cast("string")
                    )
                ).isin(
                    "NO",
                    "N",
                    "FALSE",
                    "0"
                ),
                F.lit(0)
            )
            .otherwise(
                F.lit(None).cast(IntegerType())
            )
        )

        df_prepared = (
            df
            .withColumn(
                "_expires_parsed",
                parse_date_column(
                    column_name="expires",
                    end_of_month=True
                )
            )
            .withColumn(
                "_acct_open_date_parsed",
                parse_date_column(
                    column_name="acct_open_date",
                    end_of_month=False
                )
            )
        )

        invalid_expires_count = (
            df_prepared
            .filter(
                F.col("expires").isNotNull()
                & (
                    F.trim(
                        F.col("expires").cast("string")
                    ) != ""
                )
                & F.col("_expires_parsed").isNull()
            )
            .count()
        )

        invalid_open_date_count = (
            df_prepared
            .filter(
                F.col("acct_open_date").isNotNull()
                & (
                    F.trim(
                        F.col(
                            "acct_open_date"
                        ).cast("string")
                    ) != ""
                )
                & F.col(
                    "_acct_open_date_parsed"
                ).isNull()
            )
            .count()
        )

        if invalid_expires_count > 0:
            print(
                "Cảnh báo: Có "
                f"{invalid_expires_count} bản ghi "
                "không chuyển đổi được expires."
            )

        if invalid_open_date_count > 0:
            print(
                "Cảnh báo: Có "
                f"{invalid_open_date_count} bản ghi "
                "không chuyển đổi được acct_open_date."
            )

        df_mdm_cards = (
            df_prepared
            .select(
                F.col("id")
                .cast(LongType())
                .alias("card_id"),

                F.col("client_id")
                .cast(LongType())
                .alias("client_id"),

                F.upper(
                    F.trim(
                        F.col("card_type")
                    )
                )
                .cast(StringType())
                .alias("card_type"),

                F.upper(
                    F.trim(
                        F.col("card_brand")
                    )
                )
                .cast(StringType())
                .alias("card_brand"),

                F.trim(
                    F.col("card_number")
                    .cast("string")
                )
                .alias("card_number"),

                F.col("_expires_parsed")
                .alias("expires"),

                F.trim(
                    F.col("cvv")
                    .cast("string")
                )
                .alias("cvv"),

                has_chip_code.alias("has_chip"),

                F.col("num_cards_issued")
                .cast(IntegerType())
                .alias("num_cards_issued"),

                clean_money_column(
                    "credit_limit"
                )
                .alias("credit_limit"),

                F.col(
                    "_acct_open_date_parsed"
                )
                .alias("acct_open_date"),

                F.col("year_pin_last_changed")
                .cast(IntegerType())
                .alias("year_pin_last_changed"),

                F.current_timestamp()
                .alias("etl_date")
            )
            .dropDuplicates(["card_id"])
            .cache()
        )

        card_count = df_mdm_cards.count()

        print("Dữ liệu sau mapping - mdm_cards:")
        print(
            f"Số lượng thẻ sau xử lý: {card_count}"
        )

        df_mdm_cards.show(
            5,
            truncate=False
        )

        null_card_id_count = (
            df_mdm_cards
            .filter(
                F.col("card_id").isNull()
            )
            .count()
        )

        if null_card_id_count > 0:
            print(
                "Lỗi dữ liệu: Có "
                f"{null_card_id_count} bản ghi "
                "card_id null."
            )
            return False

        null_client_id_count = (
            df_mdm_cards
            .filter(
                F.col("client_id").isNull()
            )
            .count()
        )

        if null_client_id_count > 0:
            print(
                "Cảnh báo: Có "
                f"{null_client_id_count} bản ghi "
                "client_id null."
            )

        unknown_has_chip_count = (
            df_mdm_cards
            .filter(
                F.col("has_chip").isNull()
            )
            .count()
        )

        if unknown_has_chip_count > 0:
            print(
                "Cảnh báo: Có "
                f"{unknown_has_chip_count} bản ghi "
                "không xác định được has_chip."
            )

        output_path = os.path.join(
            data_path["root_path"],
            "prc",
            "mdm_cards",
            f"etl_date={etl_date}"
        )

        print(
            f"Đường dẫn đầu ra: {output_path}"
        )

        success_parquet = write_df_to_parquet(
            df=df_mdm_cards,
            parquet_path=output_path,
            mode_write="overwrite"
        )

        if not success_parquet:
            print(
                "Lỗi khi ghi dữ liệu "
                "mdm_cards vào Parquet."
            )
            return False

        print(
            "Dữ liệu mdm_cards đã được ghi "
            f"vào Parquet tại: {output_path}"
        )

        return True

    except Exception as e:
        print(
            "Lỗi trong quá trình ETL "
            f"mdm_cards: {str(e)}"
        )

        traceback.print_exc()
        return False

    finally:
        if df_mdm_cards is not None:
            try:
                df_mdm_cards.unpersist()
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
        description="_0030204_prc_mdm_card"
    )

    parser.add_argument(
        "--etl_date",
        type=str,
        required=True,
        help="etl_date theo định dạng YYYYMMDD"
    )

    args = parser.parse_args()

    success = _0030204_prc_mdm_card(
        etl_date=args.etl_date
    )

    sys.exit(0 if success else 1)