"""
Data Cleaning and Preprocessing Pipeline
-------------------------------------------

missing value handling, outlier detection, categorical encoding,
and feature normalization/standardization.
"""

import pandas as pd
import numpy as np
import os
import warnings

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# CONFIGURATION

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "Data Set For Task", "1) iris.csv"
)
OUTPUT_DIR = SCRIPT_DIR
CLEANED_CSV = os.path.join(OUTPUT_DIR, "iris_cleaned.csv")
PREPROCESSED_CSV = os.path.join(OUTPUT_DIR, "iris_preprocessed.csv")
REPORT_TXT = os.path.join(OUTPUT_DIR, "cleaning_report.txt")

# We'll collect report lines for a final summary file
report_lines = []


def log(msg=""):
    """Print and record a message."""
    print(msg)
    report_lines.append(msg)



# STEP 0: Load the Raw Data

def load_data(path):
    print(" Loading Raw Dataset")

    log(f"\n  Source: {path}")

    df = pd.read_csv(path)

    log(f"  Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"  Columns: {list(df.columns)}")
    log(f"\n  Data Types:")
    for col in df.columns:
        log(f"    {col:20s} -> {df[col].dtype}")

    log(f"\n  First 5 rows:")
    log(df.head().to_string(index=False))
    log("")
    return df



# STEP 1: Handle Missing Data

def handle_missing_data(df):
    print(" Handling Missing Data")

    df = df.copy()

    # Inspect missing values
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct,
    })

    log(f"\n  --- Missing Value Report (Original) ---")
    log(missing_report.to_string())

    total_missing = missing.sum()
    log(f"\n  Total missing values: {total_missing}")

    if total_missing == 0:
        log("  Dataset has no missing values.")
        log("  -> Simulating missing data to demonstrate imputation techniques...\n")

        # Inject ~5% missing values for demonstration purposes
        np.random.seed(42)
        n_missing = int(0.05 * df.shape[0] * (df.shape[1] - 1))  # exclude species
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        injected_count = 0
        for _ in range(n_missing):
            row = np.random.randint(0, df.shape[0])
            col = np.random.choice(numeric_cols)
            df.at[row, col] = np.nan
            injected_count += 1

        log(f"  Injected {injected_count} NaN values across numeric columns.\n")

    # -check missing values
    missing_after_inject = df.isnull().sum()
    missing_pct_after = (df.isnull().sum() / len(df) * 100).round(2)
    log(f"  --- Missing Value Report (After Injection) ---")
    missing_report2 = pd.DataFrame({
        "Missing Count": missing_after_inject,
        "Missing %": missing_pct_after,
    })
    log(missing_report2.to_string())

    # Strategy 1: Mean Imputation (for numerical columns)
    log(f"\n  --- Strategy: Mean Imputation ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    imputer = SimpleImputer(strategy="mean")
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    for col in numeric_cols:
        n_filled = missing_after_inject[col]
        if n_filled > 0:
            mean_val = df[col].mean()
            log(f"    {col}: filled {n_filled} missing values with mean = {mean_val:.4f}")

    # Strategy 2: Median Imputation (alternative demonstration)
    log(f"\n  --- Alternative Strategy: Median Imputation ---")
    imputer_median = SimpleImputer(strategy="median")
    df_median = df.copy()
    df_median[numeric_cols] = imputer_median.fit_transform(df[numeric_cols])

    for col in numeric_cols:
        n_filled = missing_after_inject[col]
        if n_filled > 0:
            median_val = df[col].median()
            log(f"    {col}: would fill {n_filled} missing values with median = {median_val:.4f}")

    # Strategy 3: Removal (rows with any missing)
    df_dropped = df.dropna()
    rows_dropped = len(df) - len(df_dropped)
    log(f"\n  --- Alternative Strategy: Row Removal ---")
    log(f"    Rows with missing data: {rows_dropped}")
    log(f"    Remaining rows after removal: {len(df_dropped)}")
    log(f"    Data loss: {rows_dropped / len(df) * 100:.1f}%")

    # Use mean-imputed data going forward
    df = df_imputed.copy()
    remaining_missing = df.isnull().sum().sum()
    log(f"\n  -> Using Mean Imputation as primary strategy.")
    log(f"  -> Remaining missing values: {remaining_missing}")
    log("")
    return df



# STEP 2: Detect and Remove Outliers

def detect_and_remove_outliers(df):
    print(" Detecting and Removing Outliers")

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # IQR Method
    log(f"\n  --- Method: Interquartile Range (IQR) ---")
    log(f"  Formula: Outlier if value < Q1 - 1.5*IQR  or  value > Q3 + 1.5*IQR\n")

    outlier_summary = {}
    total_outliers = 0
    outlier_mask = pd.Series([False] * len(df), index=df.index)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = col_outliers.sum()
        outlier_mask |= col_outliers
        total_outliers += n_outliers

        outlier_summary[col] = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers": n_outliers,
        }

        log(f"  {col}:")
        log(f"    Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
        log(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        log(f"    Outliers found: {n_outliers}")

        if n_outliers > 0:
            outlier_vals = df.loc[col_outliers, col].values
            log(f"    Outlier values: {outlier_vals}")
        log("")

    total_rows_with_outliers = outlier_mask.sum()
    log(f"  --- Outlier Summary ---")
    log(f"  Total outlier data points : {total_outliers}")
    log(f"  Total rows with outliers  : {total_rows_with_outliers}")
    log(f"  Percentage of rows        : {total_rows_with_outliers / len(df) * 100:.1f}%")

    # Remove outlier rows
    df_clean = df[~outlier_mask].reset_index(drop=True)
    log(f"\n  Rows before outlier removal: {len(df)}")
    log(f"  Rows after outlier removal : {len(df_clean)}")
    log(f"  Rows removed              : {len(df) - len(df_clean)}")

    # Z-Score Method (alternative demonstration)
    log(f"\n  --- Alternative Method: Z-Score (|z| > 3) ---")
    from scipy import stats

    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        z_outliers = (z_scores > 3).sum()
        log(f"    {col}: {z_outliers} outliers (z > 3)")

    log("")
    return df_clean



# STEP 3: Convert Categorical Variables to Numerical

def encode_categorical(df):
    print(" Converting Categorical Variables to Numerical")

    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    log(f"\n  Categorical columns found: {cat_cols}")

    if not cat_cols:
        log("  No categorical columns to encode.")
        return df

    for col in cat_cols:
        unique_vals = df[col].unique()
        log(f"\n  Column: '{col}'")
        log(f"    Unique values ({len(unique_vals)}): {list(unique_vals)}")

    # Label Encoding
    log(f"\n  --- Method 1: Label Encoding ---")
    log(f"  Maps each category to an integer.\n")

    df_label = df.copy()
    le = LabelEncoder()

    for col in cat_cols:
        df_label[f"{col}_label"] = le.fit_transform(df_label[col])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        log(f"    {col} -> {col}_label")
        log(f"    Mapping: {mapping}")

    log(f"\n  Label-encoded preview:")
    label_cols = [c for c in df_label.columns if c.endswith("_label") or c not in cat_cols]
    log(df_label[label_cols].head().to_string(index=False))

    # One-Hot Encoding
    log(f"\n  --- Method 2: One-Hot Encoding ---")
    log(f"  Creates binary columns for each category.\n")

    df_onehot = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)

    log(f"  Columns before: {list(df.columns)}")
    log(f"  Columns after : {list(df_onehot.columns)}")
    log(f"\n  One-hot encoded preview:")
    log(df_onehot.head().to_string(index=False))

    # Return label-encoded version for the final preprocessed output
    # (keeps dataset compact)
    df_result = df.copy()
    for col in cat_cols:
        le_final = LabelEncoder()
        df_result[f"{col}_encoded"] = le_final.fit_transform(df_result[col])

    log(f"\n  -> Using Label Encoding for final output (compact format).")
    log(f"  -> One-Hot Encoding demonstrated above for reference.")
    log("")
    return df_result



# STEP 4: Normalize / Standardize Numerical Data

def normalize_and_standardize(df):
    print(" Normalizing and Standardizing Numerical Data")

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude encoded columns from scaling
    feature_cols = [c for c in numeric_cols if not c.endswith("_encoded")]

    log(f"\n  Numeric feature columns: {feature_cols}")

    # Original Statistics
    log(f"\n  --- Original Feature Statistics ---")
    stats_df = df[feature_cols].describe().round(4)
    log(stats_df.to_string())

    # Min-Max Normalization (scales to [0, 1])
    log(f"\n  --- Method 1: Min-Max Normalization (range [0, 1]) ---")

    scaler_mm = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler_mm.fit_transform(df[feature_cols])

    log(f"  Formula: X_norm = (X - X_min) / (X_max - X_min)\n")
    log(f"  Normalized Statistics:")
    log(df_normalized[feature_cols].describe().round(4).to_string())

    # Standardization (Z-score: mean=0, std=1)
    log(f"\n  --- Method 2: Standardization (Z-score, mean=0, std=1) ---")

    scaler_std = StandardScaler()
    df_standardized = df.copy()
    df_standardized[feature_cols] = scaler_std.fit_transform(df[feature_cols])

    log(f"  Formula: X_std = (X - mean) / std\n")
    log(f"  Standardized Statistics:")
    log(df_standardized[feature_cols].describe().round(4).to_string())

    # Comparison Table
    log(f"\n  --- Comparison: Original vs Normalized vs Standardized ---")
    for col in feature_cols:
        log(f"\n  {col}:")
        log(f"    Original     -> min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}, std={df[col].std():.4f}")
        log(f"    Normalized   -> min={df_normalized[col].min():.4f}, max={df_normalized[col].max():.4f}, mean={df_normalized[col].mean():.4f}, std={df_normalized[col].std():.4f}")
        log(f"    Standardized -> min={df_standardized[col].min():.4f}, max={df_standardized[col].max():.4f}, mean={df_standardized[col].mean():.4f}, std={df_standardized[col].std():.4f}")

    # Use standardized version as final output
    log(f"\n  -> Using Standardized (Z-score) data for final output.")
    log("")

    return df_standardized



# STEP 5: Save Cleaned & Preprocessed Data

def save_outputs(df_cleaned, df_preprocessed):
    print(" Saving Outputs")

    # Save cleaned CSV (after missing data handling + outlier removal)
    df_cleaned.to_csv(CLEANED_CSV, index=False)
    log(f"\n  Cleaned data saved to:")
    log(f"    {CLEANED_CSV}")
    log(f"    Shape: {df_cleaned.shape}")

    # Save preprocessed CSV (cleaned + encoded + standardized)
    df_preprocessed.to_csv(PREPROCESSED_CSV, index=False)
    log(f"\n  Preprocessed data saved to:")
    log(f"    {PREPROCESSED_CSV}")
    log(f"    Shape: {df_preprocessed.shape}")

    # Save the text report
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\n  Cleaning report saved to:")
    log(f"    {REPORT_TXT}")

    log(f"\n{'=' * 70}")
    log("Process Complete: Data Cleaning and Preprocessing")
    log(f"{'=' * 70}")



# MAIN EXECUTION

def main():

    log("  TASK 2: DATA CLEANING AND PREPROCESSING")

    log("=" * 70 + "\n")

    # Step 0: Load raw data
    df_raw = load_data(DATA_PATH)

    # Step 1: Handle missing data
    df_no_missing = handle_missing_data(df_raw)

    # Step 2: Detect and remove outliers
    df_no_outliers = detect_and_remove_outliers(df_no_missing)

    # Save the "cleaned" version (missing handled + outliers removed)
    df_cleaned = df_no_outliers.copy()

    # Step 3: Encode categorical variables
    df_encoded = encode_categorical(df_no_outliers)

    # Step 4: Normalize / standardize numerical data
    df_final = normalize_and_standardize(df_encoded)

    # Step 5: Save all outputs
    save_outputs(df_cleaned, df_final)


if __name__ == "__main__":
    main()




