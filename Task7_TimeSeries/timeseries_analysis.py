"""
Time Series Analysis and Forecasting
--------------------------------------
Decomposing, smoothing, and forecasting a time series using
classical methods (moving average, exponential smoothing, ARIMA).

Uses the classic AirPassengers dataset (monthly airline passenger
numbers from 1949-1960).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def load_air_passengers():
    """Load the classic air passengers dataset from statsmodels."""
    from statsmodels.datasets import get_rdataset
    data = get_rdataset("AirPassengers").data

    # build proper datetime index
    dates = pd.date_range(start="1949-01", periods=len(data), freq="MS")
    ts = pd.Series(data["value"].values, index=dates, name="Passengers")
    return ts


def main():
    # ── Load data ──────────────────────────────────────────────────
    print("Loading AirPassengers dataset...")
    ts = load_air_passengers()
    print(f"  Period: {ts.index[0].strftime('%Y-%m')} to {ts.index[-1].strftime('%Y-%m')}")
    print(f"  Observations: {len(ts)}")
    print(f"  Mean: {ts.mean():.1f}, Std: {ts.std():.1f}")

    # quick plot of raw data
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, color="#2c3e50", linewidth=1.5)
    ax.set_title("Monthly Airline Passengers (1949-1960)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Passengers (thousands)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_raw_timeseries.png"))
    plt.close()
    print("  Saved raw time series plot")

    # ── Decomposition ──────────────────────────────────────────────
    # multiplicative decomposition makes more sense here since the
    # seasonal amplitude grows with the level
    print("\nDecomposing time series (multiplicative)...")
    decomp = seasonal_decompose(ts, model="multiplicative", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    components = [
        (decomp.observed, "Observed", "#2c3e50"),
        (decomp.trend, "Trend", "#e74c3c"),
        (decomp.seasonal, "Seasonal", "#27ae60"),
        (decomp.resid, "Residual", "#8e44ad"),
    ]
    for ax, (comp, title, color) in zip(axes, components):
        ax.plot(comp, color=color, linewidth=1.2)
        ax.set_ylabel(title)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_decomposition.png"))
    plt.close()
    print("  Saved decomposition plot")

    # ── Stationarity check ─────────────────────────────────────────
    print("\nStationarity test (Augmented Dickey-Fuller):")
    adf_result = adfuller(ts, autolag="AIC")
    print(f"  ADF Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print(f"  Stationary: {'Yes' if adf_result[1] < 0.05 else 'No (need differencing)'}")

    # also test differenced series
    ts_diff = ts.diff().dropna()
    adf_diff = adfuller(ts_diff, autolag="AIC")
    print(f"\n  After 1st differencing:")
    print(f"  ADF Statistic: {adf_diff[0]:.4f}")
    print(f"  p-value: {adf_diff[1]:.6f}")
    print(f"  Stationary: {'Yes' if adf_diff[1] < 0.05 else 'No'}")

    # ── Train/test split ───────────────────────────────────────────
    # use last 24 months as test set
    train = ts[:-24]
    test = ts[-24:]
    print(f"\nTrain: {len(train)} months, Test: {len(test)} months")

    # ── Moving Average ─────────────────────────────────────────────
    print("\nApplying moving averages...")
    ma_6 = ts.rolling(window=6).mean()
    ma_12 = ts.rolling(window=12).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts, label="Original", alpha=0.5, color="#bdc3c7")
    ax.plot(ma_6, label="6-month MA", color="#e74c3c", linewidth=2)
    ax.plot(ma_12, label="12-month MA", color="#2980b9", linewidth=2)
    ax.set_title("Moving Average Smoothing")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_moving_averages.png"))
    plt.close()
    print("  Saved moving average plot")

    # ── Exponential Smoothing (Holt-Winters) ───────────────────────
    print("Fitting Holt-Winters exponential smoothing...")
    hw_model = ExponentialSmoothing(
        train, trend="mul", seasonal="mul", seasonal_periods=12
    ).fit(optimized=True)

    hw_forecast = hw_model.forecast(24)
    hw_rmse = np.sqrt(mean_squared_error(test, hw_forecast))
    hw_mae = mean_absolute_error(test, hw_forecast)
    print(f"  Holt-Winters RMSE: {hw_rmse:.2f}")
    print(f"  Holt-Winters MAE: {hw_mae:.2f}")

    # ── ARIMA Model ────────────────────────────────────────────────
    # using ARIMA(1,1,1)x(1,1,1,12) — a reasonable starting point
    # for monthly data with yearly seasonality
    print("\nFitting ARIMA(1,1,1) model...")
    arima_model = ARIMA(train, order=(1, 1, 1)).fit()
    arima_forecast = arima_model.forecast(24)
    arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
    arima_mae = mean_absolute_error(test, arima_forecast)
    print(f"  ARIMA RMSE: {arima_rmse:.2f}")
    print(f"  ARIMA MAE: {arima_mae:.2f}")

    # try SARIMA for better seasonality handling
    print("Fitting SARIMA(1,1,1)(1,1,1,12) model...")
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sarima_model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)
    ).fit(disp=False)
    sarima_forecast = sarima_model.forecast(24)
    sarima_rmse = np.sqrt(mean_squared_error(test, sarima_forecast))
    sarima_mae = mean_absolute_error(test, sarima_forecast)
    print(f"  SARIMA RMSE: {sarima_rmse:.2f}")
    print(f"  SARIMA MAE: {sarima_mae:.2f}")

    # ── Forecast comparison plot ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train, label="Training Data", color="#2c3e50", linewidth=1.2)
    ax.plot(test, label="Actual (Test)", color="#2c3e50", linewidth=2, linestyle="--")
    ax.plot(hw_forecast, label=f"Holt-Winters (RMSE={hw_rmse:.1f})",
            color="#e74c3c", linewidth=2)
    ax.plot(arima_forecast, label=f"ARIMA (RMSE={arima_rmse:.1f})",
            color="#3498db", linewidth=2)
    ax.plot(sarima_forecast, label=f"SARIMA (RMSE={sarima_rmse:.1f})",
            color="#27ae60", linewidth=2)

    ax.axvline(x=test.index[0], color="gray", linestyle=":", alpha=0.7)
    ax.set_title("Forecast Comparison (24-month horizon)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_forecast_comparison.png"))
    plt.close()
    print("  Saved forecast comparison plot")

    # ── Model comparison ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    models_names = ["Holt-Winters", "ARIMA(1,1,1)", "SARIMA"]
    rmses = [hw_rmse, arima_rmse, sarima_rmse]
    maes = [hw_mae, arima_mae, sarima_mae]

    x = np.arange(len(models_names))
    width = 0.35
    ax.bar(x - width/2, rmses, width, label="RMSE", color="#e74c3c", alpha=0.8)
    ax.bar(x + width/2, maes, width, label="MAE", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models_names)
    ax.set_ylabel("Error")
    ax.set_title("Model Performance Comparison")
    ax.legend()

    for i, (r, m) in enumerate(zip(rmses, maes)):
        ax.text(i - width/2, r + 0.5, f"{r:.1f}", ha="center", fontsize=10)
        ax.text(i + width/2, m + 0.5, f"{m:.1f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_model_comparison.png"))
    plt.close()
    print("  Saved model comparison plot")

    # ── Summary ────────────────────────────────────────────────────

    print("TIME SERIES ANALYSIS SUMMARY")

    print(f"  Best model: {'SARIMA' if sarima_rmse < hw_rmse else 'Holt-Winters'}")
    print(f"\n  Model Performance:")
    print(f"    Holt-Winters : RMSE={hw_rmse:.2f}, MAE={hw_mae:.2f}")
    print(f"    ARIMA(1,1,1) : RMSE={arima_rmse:.2f}, MAE={arima_mae:.2f}")
    print(f"    SARIMA       : RMSE={sarima_rmse:.2f}, MAE={sarima_mae:.2f}")
    print(f"\n  Key observations:")
    print(f"    - Clear upward trend with multiplicative seasonality")
    print(f"    - Peak travel in July-August each year")
    print(f"    - SARIMA captures seasonal patterns better than plain ARIMA")

    # save report
    with open(os.path.join(SCRIPT_DIR, "timeseries_report.txt"), "w") as f:
        f.write("Time Series Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: AirPassengers (1949-1960)\n")
        f.write(f"Train: {len(train)} months, Test: {len(test)} months\n\n")
        f.write("Model Results:\n")
        f.write(f"  Holt-Winters: RMSE={hw_rmse:.2f}, MAE={hw_mae:.2f}\n")
        f.write(f"  ARIMA(1,1,1): RMSE={arima_rmse:.2f}, MAE={arima_mae:.2f}\n")
        f.write(f"  SARIMA(1,1,1)(1,1,1,12): RMSE={sarima_rmse:.2f}, MAE={sarima_mae:.2f}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()




