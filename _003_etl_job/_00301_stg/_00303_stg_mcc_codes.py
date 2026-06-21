import sys
import os
import traceback
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.abspath(
    os.path.join(current_dir, "..", "..")
)

sys.path.insert(0, config_path)

from _002_utils._00201_utils import (
    create_spark_session,
    read_df_from_table,
    read_df_from_parquet,
    write_df_to_parquet
)
from _001_config._00102_path_config import data_path


def _00303_stg_mcc_codes(etl_date: str):
    spark = None

    try:
        spark = create_spark_session(
            "_00303_stg_mcc_codes"
        )

        script_sql = """
            SELECT *
            FROM banking.mcc_codes
        """

        df = read_df_from_table(
            spark=spark,
            script_sql=script_sql
        )

        if df is None:
            print("Không thể đọc dữ liệu từ SQL Server.")
            return False

        print("Dữ liệu đọc từ bảng mcc_codes:")
        df.show(5, truncate=False)

        source_count = df.count()
        print(f"Số lượng bản ghi nguồn: {source_count}")

        output_path = os.path.join(
            data_path["root_path"],
            "stg",
            "mcc_codes",
            f"etl_date={etl_date}"
        )

        success_parquet = write_df_to_parquet(
            df=df,
            parquet_path=output_path,
            mode_write="overwrite"
        )

        if not success_parquet:
            print("Lỗi khi ghi dữ liệu vào Parquet.")
            return False

        print(
            f"Dữ liệu đã được ghi vào Parquet tại: "
            f"{output_path}"
        )

        df_parquet = read_df_from_parquet(
            spark=spark,
            parquet_path=output_path
        )

        if df_parquet is None:
            print("Không thể đọc lại dữ liệu từ Parquet.")
            return False

        print("Dữ liệu đọc lại từ Parquet:")
        df_parquet.show(5, truncate=False)

        parquet_count = df_parquet.count()

        print(
            f"Số lượng bản ghi sau khi đọc lại: "
            f"{parquet_count}"
        )

        if source_count != parquet_count:
            print(
                "Cảnh báo: Số lượng bản ghi nguồn và "
                "Parquet không bằng nhau."
            )
            return False

        print("ETL hoàn thành thành công.")
        return True

    except Exception as e:
        print(f"Lỗi trong quá trình ETL: {str(e)}")
        traceback.print_exc()
        return False

    finally:
        if spark is not None:
            spark.stop()
            print("Đã đóng SparkSession.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="_00303_stg_mcc_codes"
    )

    parser.add_argument(
        "--etl_date",
        type=str,
        required=True,
        help="etl_date theo định dạng YYYYMMDD"
    )

    args = parser.parse_args()

    success = _00303_stg_mcc_codes(
        etl_date=args.etl_date
    )

    sys.exit(0 if success else 1)