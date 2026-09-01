"""
Predictive Modeling - House Price Regression
----------------------------------------------
Builds and compares multiple regression models to predict
median house values using the Boston Housing dataset.

Models: Linear, Ridge, Lasso, Decision Tree, Random Forest,
        Gradient Boosting, SVR
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# CONFIGURATION

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "Data Set For Task", "4) house Prediction Data Set.csv"
)
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
REPORT_FILE = os.path.join(SCRIPT_DIR, "regression_report.txt")

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# Boston Housing feature names
FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]
TARGET_NAME = "MEDV"

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)



# STEP 0: Load and Explore Data

def load_data():
    print(" Loading and Exploring Dataset")

    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, engine="python")
    df.columns = FEATURE_NAMES + [TARGET_NAME]

    log(f"\n  Source : {DATA_PATH}")
    log(f"  Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"  Features: {FEATURE_NAMES}")
    log(f"  Target  : {TARGET_NAME} (Median value of homes in $1000s)")

    log(f"\n  --- Feature Descriptions ---")
    descriptions = {
        "CRIM": "Per capita crime rate by town",
        "ZN": "Proportion of residential land zoned for lots > 25,000 sq.ft",
        "INDUS": "Proportion of non-retail business acres per town",
        "CHAS": "Charles River dummy variable (1 if borders river)",
        "NOX": "Nitric oxides concentration (parts per 10 million)",
        "RM": "Average number of rooms per dwelling",
        "AGE": "Proportion of owner-occupied units built prior to 1940",
        "DIS": "Weighted distances to five Boston employment centres",
        "RAD": "Index of accessibility to radial highways",
        "TAX": "Full-value property-tax rate per $10,000",
        "PTRATIO": "Pupil-teacher ratio by town",
        "B": "1000(Bk - 0.63)^2 where Bk is the proportion of Black residents",
        "LSTAT": "% lower status of the population",
        "MEDV": "Median value of owner-occupied homes in $1000s (TARGET)",
    }
    for feat, desc in descriptions.items():
        log(f"    {feat:8s}: {desc}")

    log(f"\n  --- Missing Values ---")
    log(f"    Total: {df.isnull().sum().sum()}")

    log(f"\n  --- Target Variable (MEDV) ---")
    log(f"    Mean  : ${df[TARGET_NAME].mean():.2f}k")
    log(f"    Median: ${df[TARGET_NAME].median():.2f}k")
    log(f"    Std   : ${df[TARGET_NAME].std():.2f}k")
    log(f"    Range : ${df[TARGET_NAME].min():.2f}k - ${df[TARGET_NAME].max():.2f}k")
    log("")

    return df



# STEP 1: Train-Test Split

def split_data(df):
    print(" Splitting Data into Training and Testing Sets")

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log(f"\n  Split ratio: 80% train / 20% test")
    log(f"  Random state: 42 (for reproducibility)")
    log(f"\n  Training set: {X_train.shape[0]} samples")
    log(f"  Testing set : {X_test.shape[0]} samples")
    log(f"\n  Training target mean : ${y_train.mean():.2f}k")
    log(f"  Testing target mean  : ${y_test.mean():.2f}k")

    # Feature scaling for models that need it
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_NAMES, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_NAMES, index=X_test.index
    )

    log(f"\n  Features standardized (StandardScaler) for scale-sensitive models.")
    log("")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler



# STEP 2: Train Linear Regression

def train_linear_regression(X_train, X_test, y_train, y_test):
    print(" Training Linear Regression Model")

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Coefficients
    log(f"\n  --- Model Coefficients ---")
    log(f"  Intercept: {model.intercept_:.4f}")
    log(f"\n  Feature Coefficients:")
    coef_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Coefficient": model.coef_,
        "Abs_Coeff": np.abs(model.coef_),
    }).sort_values("Abs_Coeff", ascending=False)

    for _, row in coef_df.iterrows():
        direction = "+" if row["Coefficient"] > 0 else "-"
        log(f"    {row['Feature']:8s}: {direction}{abs(row['Coefficient']):.4f}")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    log(f"\n  --- Training Performance ---")
    log(f"    R-squared : {r2_score(y_train, y_pred_train):.4f}")
    log(f"    MSE       : {mean_squared_error(y_train, y_pred_train):.4f}")
    log(f"    RMSE      : {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    log(f"    MAE       : {mean_absolute_error(y_train, y_pred_train):.4f}")

    log(f"\n  --- Testing Performance ---")
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    log(f"    R-squared : {r2:.4f}")
    log(f"    MSE       : {mse:.4f}")
    log(f"    RMSE      : {rmse:.4f}")
    log(f"    MAE       : {mae:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    log(f"\n  --- 5-Fold Cross-Validation (R-squared) ---")
    log(f"    Scores: {[f'{s:.4f}' for s in cv_scores]}")
    log(f"    Mean  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    log("")

    return model, y_pred_test



# STEP 3: Evaluate with Metrics & Visualizations

def evaluate_and_visualize(model, X_test, y_test, y_pred_test, X_train, y_train):
    print(" Model Evaluation Visualizations")

    # Actual vs Predicted
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Linear Regression - Model Evaluation", fontsize=16, fontweight="bold", y=1.02)

    ax = axes[0]
    ax.scatter(y_test, y_pred_test, alpha=0.6, c="#3498db", edgecolors="white", s=50)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual MEDV ($1000s)", fontsize=12)
    ax.set_ylabel("Predicted MEDV ($1000s)", fontsize=12)
    ax.set_title("Actual vs Predicted", fontsize=13)
    ax.legend()

    # Residual Plot
    residuals = y_test - y_pred_test
    ax = axes[1]
    ax.scatter(y_pred_test, residuals, alpha=0.6, c="#e74c3c", edgecolors="white", s=50)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted MEDV ($1000s)", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title("Residual Plot", fontsize=13)

    # Residual Distribution
    ax = axes[2]
    ax.hist(residuals, bins=20, color="#2ecc71", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Residual Distribution", fontsize=13)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_linear_regression_eval.png")
    plt.savefig(path)
    plt.close()
    log(f"\n  Saved: {path}")

    # Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    coef_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient")

    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="white")
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("Linear Regression - Feature Coefficients", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_feature_coefficients.png")
    plt.savefig(path)
    plt.close()
    log(f"  Saved: {path}")
    log("")



# STEP 4: Multiple Models Comparison

def compare_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    print(" Comparing Multiple Regression Models")

    models = {
        "Linear Regression": (LinearRegression(), X_train, X_test),
        "Ridge Regression": (Ridge(alpha=1.0), X_train, X_test),
        "Lasso Regression": (Lasso(alpha=0.1), X_train, X_test),
        "Decision Tree": (DecisionTreeRegressor(random_state=42, max_depth=10), X_train, X_test),
        "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, random_state=42), X_train, X_test),
        "SVR (RBF)": (SVR(kernel="rbf", C=10, gamma="scale"), X_train_scaled, X_test_scaled),
    }

    results = []

    for name, (model, X_tr, X_te) in models.items():
        log(f"\n  Training: {name}...")

        model.fit(X_tr, y_train)
        y_pred_train = model.predict(X_tr)
        y_pred_test = model.predict(X_te)

        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            "Model": name,
            "Train R2": train_r2,
            "Test R2": test_r2,
            "Test MSE": test_mse,
            "Test RMSE": test_rmse,
            "Test MAE": test_mae,
            "CV R2 Mean": cv_mean,
            "CV R2 Std": cv_std,
            "Predictions": y_pred_test,
            "Model_Obj": model,
        })

        log(f"    Train R2={train_r2:.4f} | Test R2={test_r2:.4f} | RMSE={test_rmse:.4f} | CV R2={cv_mean:.4f}(+/-{cv_std:.4f})")

    # Table
    results_df = pd.DataFrame(results)
    log(f"\n  {'=' * 70}")
    log(f"  MODEL COMPARISON RESULTS")
    log(f"  {'=' * 70}")

    display_cols = ["Model", "Train R2", "Test R2", "Test RMSE", "Test MAE", "CV R2 Mean"]
    display_df = results_df[display_cols].copy()
    display_df = display_df.sort_values("Test R2", ascending=False)
    for col in display_cols[1:]:
        display_df[col] = display_df[col].round(4)
    log(f"\n{display_df.to_string(index=False)}")

    # Best model
    best_idx = results_df["Test R2"].idxmax()
    best = results_df.iloc[best_idx]
    log(f"\n  BEST MODEL: {best['Model']}")
    log(f"    Test R-squared: {best['Test R2']:.4f}")
    log(f"    Test RMSE     : {best['Test RMSE']:.4f}")
    log(f"    CV R2         : {best['CV R2 Mean']:.4f} (+/- {best['CV R2 Std']:.4f})")

    # : Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold", y=1.02)

    # R2 comparison
    ax = axes[0]
    sorted_df = results_df.sort_values("Test R2", ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_df)))
    ax.barh(sorted_df["Model"], sorted_df["Test R2"], color=colors, edgecolor="white")
    ax.set_xlabel("R-squared (Test Set)", fontsize=12)
    ax.set_title("R-squared Comparison", fontsize=13, fontweight="bold")
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax.text(row["Test R2"] + 0.01, i, f'{row["Test R2"]:.4f}', va="center", fontsize=10)

    # RMSE comparison
    ax = axes[1]
    sorted_df2 = results_df.sort_values("Test RMSE", ascending=False)
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_df2)))
    ax.barh(sorted_df2["Model"], sorted_df2["Test RMSE"], color=colors2, edgecolor="white")
    ax.set_xlabel("RMSE (Test Set)", fontsize=12)
    ax.set_title("RMSE Comparison (lower is better)", fontsize=13, fontweight="bold")
    for i, (_, row) in enumerate(sorted_df2.iterrows()):
        ax.text(row["Test RMSE"] + 0.05, i, f'{row["Test RMSE"]:.4f}', va="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_model_comparison.png")
    plt.savefig(path)
    plt.close()
    log(f"\n  Saved: {path}")

    # vs Predicted for all models
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Actual vs Predicted - All Models", fontsize=16, fontweight="bold", y=1.02)
    axes = axes.flatten()

    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

    for idx, res in enumerate(results):
        ax = axes[idx]
        ax.scatter(y_test, res["Predictions"], alpha=0.6, c=palette[idx % len(palette)],
                   edgecolors="white", s=40)
        min_v = min(y_test.min(), res["Predictions"].min())
        max_v = max(y_test.max(), res["Predictions"].max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5)
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{res['Model']}\nR2={res['Test R2']:.4f}", fontsize=11)

    # Hide empty subplot
    axes[-1].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_all_models_predictions.png")
    plt.savefig(path)
    plt.close()
    log(f"  Saved: {path}")

    # Importance (Random Forest)
    rf_model = results_df[results_df["Model"] == "Random Forest"]["Model_Obj"].values[0]
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_imp = plt.cm.viridis(np.linspace(0.2, 0.9, len(imp_df)))
    ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors_imp, edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Random Forest - Feature Importance", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "05_rf_feature_importance.png")
    plt.savefig(path)
    plt.close()
    log(f"  Saved: {path}")

    log(f"\n  --- Random Forest Feature Importance ---")
    for _, row in imp_df.sort_values("Importance", ascending=False).iterrows():
        bar = "#" * int(row["Importance"] * 50)
        log(f"    {row['Feature']:8s}: {row['Importance']:.4f} {bar}")

    log("")
    return results_df



# STEP 5: Final Summary

def generate_summary(results_df):
    print(" Final Summary")

    log(f"\n  --- Key Findings ---")
    log(f"\n  1. LINEAR REGRESSION BASELINE:")
    lr = results_df[results_df["Model"] == "Linear Regression"].iloc[0]
    log(f"     - Achieves R2 = {lr['Test R2']:.4f} on the test set")
    log(f"     - RMSE of ${lr['Test RMSE']:.2f}k (average prediction error)")
    log(f"     - Reasonable baseline but limited by linearity assumption")

    log(f"\n  2. BEST PERFORMING MODEL:")
    best = results_df.loc[results_df["Test R2"].idxmax()]
    log(f"     - {best['Model']} with R2 = {best['Test R2']:.4f}")
    log(f"     - RMSE = ${best['Test RMSE']:.2f}k")
    log(f"     - {((best['Test R2'] - lr['Test R2']) / lr['Test R2'] * 100):.1f}% improvement over Linear Regression")

    log(f"\n  3. MODEL RANKINGS (by Test R2):")
    ranked = results_df.sort_values("Test R2", ascending=False)
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        log(f"     {rank}. {row['Model']:20s} R2={row['Test R2']:.4f}  RMSE={row['Test RMSE']:.4f}")

    log(f"\n  4. KEY FEATURES FOR PREDICTION:")
    log(f"     - LSTAT (% lower status population) - strongest predictor")
    log(f"     - RM (average rooms per dwelling) - strong positive predictor")
    log(f"     - DIS (distance to employment centers) - important spatial factor")

    log(f"\n  5. GENERATED OUTPUTS:")
    log(f"     - 01_linear_regression_eval.png  (actual vs predicted, residuals)")
    log(f"     - 02_feature_coefficients.png    (linear regression coefficients)")
    log(f"     - 03_model_comparison.png        (R2 and RMSE bar charts)")
    log(f"     - 04_all_models_predictions.png  (scatter plots for all models)")
    log(f"     - 05_rf_feature_importance.png   (Random Forest importances)")

    log(f"\n{'=' * 70}")
    log("Level 2 Process Complete: Predictive Modeling (Regression)")
    log(f"{'=' * 70}")



# MAIN

def main():

    log("  LEVEL 2, TASK 1: PREDICTIVE MODELING (REGRESSION)")

    log("=" * 70 + "\n")

    # Step 0: Load data
    df = load_data()

    # Step 1: Split data
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = split_data(df)

    # Step 2: Train linear regression
    lr_model, y_pred_lr = train_linear_regression(X_train, X_test, y_train, y_test)

    # Step 3: Evaluate and visualize
    evaluate_and_visualize(lr_model, X_test, y_test, y_pred_lr, X_train, y_train)

    # Step 4: Compare multiple models
    results_df = compare_models(X_train, X_test, y_train, y_test, X_train_s, X_test_s)

    # Step 5: Summary
    generate_summary(results_df)

    # Save report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()




