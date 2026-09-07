import os
from utils.m4_summary import M4Summary


def main():
    # 这里改成你的预测结果目录
    forecast_dir = "./m4_results/o/"

    # 这里改成你的 M4 数据集根目录
    root_path = r"D:\TimeSeriesForecasting\AutoTimes-main\Timeseriesdata/m4"

    required = {
        "Weekly_forecast.csv",
        "Monthly_forecast.csv",
        "Yearly_forecast.csv",
        "Daily_forecast.csv",
        "Hourly_forecast.csv",
        "Quarterly_forecast.csv",
    }

    existing = set(os.listdir(forecast_dir))
    print("Existing files:", existing)

    missing = required - existing
    if len(missing) > 0:
        raise FileNotFoundError(f"Missing forecast files: {missing}")

    evaluator = M4Summary(forecast_dir, root_path)
    smape_results, owa_results, mape, mase = evaluator.evaluate()

    print("SMAPE:")
    print(smape_results)

    print("MAPE:")
    print(mape)

    print("MASE:")
    print(mase)

    print("OWA:")
    print(owa_results)


if __name__ == "__main__":
    main()