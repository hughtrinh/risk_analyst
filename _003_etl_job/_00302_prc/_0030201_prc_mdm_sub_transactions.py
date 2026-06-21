import sys
import os
import traceback
import argparse

from pyspark.sql import functions as F
from pyspark.sql.types import (
    LongType,
    StringType,
    TimestampType,
    DecimalType
)

current_dir = os.path.dirname(os.path.abspath(__file__))

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


def _0030201_prc_mdm_sub_transactions(etl_date: str):
    spark = None

    try:
        spark = create_spark_session(
            "_0030201_prc_mdm_sub_transactions"
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
                f"Không thể đọc dữ liệu từ Parquet: {input_path}"
            )
            return False

        print("Dữ liệu đọc từ Parquet - stg transactions:")
        print(
            f"Số lượng bản ghi từ stg transactions: "
            f"{df.count()}"
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

        df_mdm_sub_transactions = df.select(
            F.col("id")
                .cast(LongType())
                .alias("transactions_id"),

            F.col("date")
                .cast(TimestampType())
                .alias("transaction_datetime"),

            F.col("client_id")
                .cast(LongType())
                .alias("client_id"),

            F.col("card_id")
                .cast(LongType())
                .alias("card_id"),

            F.col("amount")
                .cast(DecimalType(18, 2))
                .alias("transaction_amount"),

            F.when(
                F.col("use_chip") == "Chip Transaction",
                F.lit("CHIP")
            )
            .when(
                F.col("use_chip") == "Swipe Transaction",
                F.lit("SWIPE")
            )
            .when(
                F.col("use_chip") == "Online Transaction",
                F.lit("ONLINE")
            )
            .otherwise(F.lit("UNKNOWN"))
            .alias("transaction_method"),

            merchant_key.alias("merchant_key"),

            F.col("merchant_id")
                .cast(LongType())
                .alias("merchant_id"),

            F.col("mcc")
                .cast(StringType())
                .alias("mcc"),

            F.col("errors")
                .cast(StringType())
                .alias("errors"),

            F.current_timestamp()
                .alias("etl_date")
        )

        print("Dữ liệu mdm_sub_transactions sau mapping:")
        df_mdm_sub_transactions.show(5, truncate=False)

        print(
            f"Số lượng bản ghi mdm_sub_transactions: "
            f"{df_mdm_sub_transactions.count()}"
        )

        output_path = os.path.join(
            data_path["root_path"],
            "prc",
            "mdm_sub_transactions",
            f"etl_date={etl_date}"
        )

        success_parquet = write_df_to_parquet(
            df=df_mdm_sub_transactions,
            parquet_path=output_path,
            mode_write="overwrite"
        )

        if not success_parquet:
            print(
                "Lỗi khi ghi dữ liệu "
                "mdm_sub_transactions vào Parquet."
            )
            return False

        print(
            "Dữ liệu mdm_sub_transactions đã được ghi "
            f"vào Parquet tại: {output_path}"
        )

        return True

    except Exception as e:
        print(
            "Lỗi trong quá trình ETL "
            f"mdm_sub_transactions: {str(e)}"
        )
        traceback.print_exc()
        return False

    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="_0030201_prc_mdm_sub_transactions"
    )

    parser.add_argument(
        "--etl_date",
        type=str,
        required=True,
        help="etl_date (YYYYMMDD)"
    )

    args = parser.parse_args()

    _0030201_prc_mdm_sub_transactions(
        etl_date=args.etl_date
    )