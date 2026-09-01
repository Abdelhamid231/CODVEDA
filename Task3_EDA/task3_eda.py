"""
Exploratory Data Analysis on the Iris Dataset
------------------------------------------------
Computes summary statistics, creates various visualizations,
and builds correlation matrices to understand the data structure.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os


# CONFIGURATION

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "Data Set For Task", "1) iris.csv"
)
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
REPORT_FILE = os.path.join(SCRIPT_DIR, "eda_report.txt")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Style configuration
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
    "font.family": "sans-serif",
})

SPECIES_COLORS = {"setosa": "#2ecc71", "versicolor": "#3498db", "virginica": "#e74c3c"}

# Report lines collector
report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)



# STEP 0: Load Data

def load_data():
    print(" Loading Dataset")

    df = pd.read_csv(DATA_PATH)
    log(f"\n  Source : {DATA_PATH}")
    log(f"  Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"  Columns: {list(df.columns)}")
    log(f"  Species: {list(df['species'].unique())}")
    log(f"  Samples per species:")
    for species, count in df["species"].value_counts().items():
        log(f"    {species:15s}: {count}")
    log("")
    return df



# STEP 1: Summary Statistics

def compute_summary_statistics(df):
    print(" Summary Statistics")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Overall descriptive statistics
    log("\n  --- Overall Descriptive Statistics ---")
    desc = df[numeric_cols].describe().round(4)
    log(desc.to_string())

    # Additional statistics
    log("\n  --- Additional Statistics ---")
    stats_table = pd.DataFrame(index=numeric_cols)
    stats_table["Mean"] = df[numeric_cols].mean().round(4)
    stats_table["Median"] = df[numeric_cols].median().round(4)
    stats_table["Variance"] = df[numeric_cols].var().round(4)
    stats_table["Std Dev"] = df[numeric_cols].std().round(4)
    stats_table["Skewness"] = df[numeric_cols].skew().round(4)
    stats_table["Kurtosis"] = df[numeric_cols].kurtosis().round(4)
    stats_table["Range"] = (df[numeric_cols].max() - df[numeric_cols].min()).round(4)
    stats_table["IQR"] = (df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)).round(4)
    log(stats_table.to_string())

    # Statistics per species
    log("\n  --- Statistics Grouped by Species ---")
    for species in df["species"].unique():
        log(f"\n  >> {species.upper()}")
        species_data = df[df["species"] == species][numeric_cols]
        species_stats = pd.DataFrame({
            "Mean": species_data.mean().round(4),
            "Median": species_data.median().round(4),
            "Std": species_data.std().round(4),
            "Min": species_data.min().round(4),
            "Max": species_data.max().round(4),
        })
        log(species_stats.to_string())

    log("")
    return stats_table



# STEP 2: Data Visualizations

def plot_histograms(df):
    """a: Histograms showing distribution of each feature."""
    log("  [2a] Generating histograms...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribution of Iris Features (Histograms)", fontsize=16, fontweight="bold", y=1.02)

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 2][idx % 2]
        for species in df["species"].unique():
            data = df[df["species"] == species][col]
            ax.hist(data, bins=15, alpha=0.6, label=species,
                    color=SPECIES_COLORS[species], edgecolor="white", linewidth=0.5)
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_histograms.png")
    plt.savefig(path)
    plt.close()
    log(f"    Saved: {path}")


def plot_boxplots(df):
    """b: Box plots for outlier visualization."""
    log("  [2b] Generating box plots...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Distributions by Species (Box Plots)", fontsize=16, fontweight="bold", y=1.02)

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 2][idx % 2]
        sns.boxplot(data=df, x="species", y=col, ax=ax,
                    palette=SPECIES_COLORS, width=0.5, linewidth=1.2,
                    flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "#e74c3c"})
        ax.set_xlabel("Species", fontsize=12)
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"{col.replace('_', ' ').title()} by Species", fontsize=13)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_boxplots.png")
    plt.savefig(path)
    plt.close()
    log(f"    Saved: {path}")


def plot_scatter_plots(df):
    """c: Scatter plots for feature relationships."""
    log("  [2c] Generating scatter plots...")

    # Key scatter plots
    scatter_pairs = [
        ("sepal_length", "sepal_width"),
        ("petal_length", "petal_width"),
        ("sepal_length", "petal_length"),
        ("sepal_width", "petal_width"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Relationships (Scatter Plots)", fontsize=16, fontweight="bold", y=1.02)

    for idx, (x_col, y_col) in enumerate(scatter_pairs):
        ax = axes[idx // 2][idx % 2]
        for species in df["species"].unique():
            mask = df["species"] == species
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                       c=SPECIES_COLORS[species], label=species,
                       alpha=0.7, s=50, edgecolors="white", linewidth=0.5)
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}", fontsize=13)
        ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_scatter_plots.png")
    plt.savefig(path)
    plt.close()
    log(f"    Saved: {path}")


def plot_pairplot(df):
    """Full pair plot showing all feature combinations."""
    log("  [2d] Generating pair plot...")

    g = sns.pairplot(df, hue="species", palette=SPECIES_COLORS,
                     diag_kind="kde", plot_kws={"alpha": 0.6, "s": 40, "edgecolor": "white"},
                     height=2.5, aspect=1)
    g.figure.suptitle("Iris Dataset — Pair Plot", fontsize=16, fontweight="bold", y=1.02)

    path = os.path.join(PLOTS_DIR, "04_pairplot.png")
    g.savefig(path)
    plt.close()
    log(f"    Saved: {path}")


def plot_violin_plots(df):
    """Violin plots combining box plot + KDE."""
    log("  [2e] Generating violin plots...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Distributions by Species (Violin Plots)", fontsize=16, fontweight="bold", y=1.02)

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 2][idx % 2]
        sns.violinplot(data=df, x="species", y=col, ax=ax,
                       palette=SPECIES_COLORS, inner="quartile", linewidth=1.2)
        ax.set_xlabel("Species", fontsize=12)
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"{col.replace('_', ' ').title()} by Species", fontsize=13)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "05_violin_plots.png")
    plt.savefig(path)
    plt.close()
    log(f"    Saved: {path}")


def create_visualizations(df):
    print(" Data Visualizations")

    log("")

    plot_histograms(df)
    plot_boxplots(df)
    plot_scatter_plots(df)
    plot_pairplot(df)
    plot_violin_plots(df)

    log(f"\n  All visualizations saved to: {PLOTS_DIR}")
    log("")



# STEP 3: Correlation Analysis

def correlation_analysis(df):
    print(" Correlation Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Pearson Correlation Matrix
    log("\n  --- Pearson Correlation Matrix ---")
    corr_matrix = df[numeric_cols].corr(method="pearson").round(4)
    log(corr_matrix.to_string())

    # Interpret strong correlations
    log("\n  --- Strong Correlations (|r| > 0.5) ---")
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.5:
                strength = "Strong" if abs(r) > 0.7 else "Moderate"
                direction = "positive" if r > 0 else "negative"
                log(f"    {numeric_cols[i]} <-> {numeric_cols[j]}: r = {r:.4f} ({strength} {direction})")

    # Spearman Rank Correlation (alternative)
    log("\n  --- Spearman Rank Correlation Matrix ---")
    spearman_corr = df[numeric_cols].corr(method="spearman").round(4)
    log(spearman_corr.to_string())

    # Correlation Heatmap
    log("\n  Generating correlation heatmap...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Correlation Matrices", fontsize=16, fontweight="bold", y=1.02)

    # Pearson
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, square=True, linewidths=1, ax=axes[0],
                vmin=-1, vmax=1,
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                cbar_kws={"shrink": 0.8})
    axes[0].set_title("Pearson Correlation", fontsize=14, fontweight="bold")

    # Spearman
    mask2 = np.triu(np.ones_like(spearman_corr, dtype=bool))
    sns.heatmap(spearman_corr, mask=mask2, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, square=True, linewidths=1, ax=axes[1],
                vmin=-1, vmax=1,
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                cbar_kws={"shrink": 0.8})
    axes[1].set_title("Spearman Rank Correlation", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "06_correlation_heatmaps.png")
    plt.savefig(path)
    plt.close()
    log(f"    Saved: {path}")

    # Per-Species Correlation Heatmaps
    log("  Generating per-species correlation heatmaps...")

    species_list = df["species"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("Correlation Matrices by Species", fontsize=16, fontweight="bold", y=1.05)

    for idx, species in enumerate(species_list):
        species_corr = df[df["species"] == species][numeric_cols].corr().round(3)
        mask_s = np.triu(np.ones_like(species_corr, dtype=bool))
        sns.heatmap(species_corr, mask=mask_s, annot=True, fmt=".3f", cmap="RdBu_r",
                    center=0, square=True, linewidths=1, ax=axes[idx],
                    vmin=-1, vmax=1,
                    annot_kws={"fontsize": 11, "fontweight": "bold"},
                    cbar_kws={"shrink": 0.8})
        axes[idx].set_title(f"{species.title()}", fontsize=14, fontweight="bold",
                            color=SPECIES_COLORS[species])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "07_correlation_by_species.png")
    plt.savefig(path)
    plt.close()
    log(f"    Saved: {path}")

    log("")
    return corr_matrix



# STEP 4: EDA Insights Report

def generate_insights_report(df, stats_table, corr_matrix):
    print(" EDA Insights Report")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    log("  EXPLORATORY DATA ANALYSIS -- KEY INSIGHTS")

    # 1: Dataset Overview
    log("\n  1. DATASET OVERVIEW")
    log(f"     - The Iris dataset contains {len(df)} samples across 3 species.")
    log(f"     - Each species has {len(df)//3} samples (perfectly balanced).")
    log(f"     - 4 numerical features: {', '.join(numeric_cols)}.")
    log(f"     - No missing values in the original dataset.")

    # 2: Feature Distributions
    log("\n  2. FEATURE DISTRIBUTIONS")
    for col in numeric_cols:
        skew = df[col].skew()
        skew_dir = "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "approximately symmetric")
        log(f"     - {col}: range [{df[col].min():.1f}, {df[col].max():.1f}], {skew_dir} (skew={skew:.3f})")

    log(f"     - Petal measurements show bimodal distributions (setosa is clearly separated).")
    log(f"     - Sepal measurements overlap more across species.")

    # 3: Species Differences
    log("\n  3. SPECIES DIFFERENCES")
    log(f"     - Setosa is easily distinguishable: smallest petals (length ~1.5, width ~0.2).")
    log(f"     - Versicolor and Virginica overlap in sepal dimensions but differ in petals.")
    log(f"     - Virginica has the largest petals overall (length ~5.6, width ~2.0).")

    # Compute per-species means for comparison
    species_means = df.groupby("species")[numeric_cols].mean()
    log(f"\n     Species Means:")
    log(species_means.round(2).to_string())

    # 4: Correlations
    log("\n  4. KEY CORRELATIONS")
    log(f"     - Petal length & petal width: r = {corr_matrix.loc['petal_length', 'petal_width']:.4f} (very strong positive)")
    log(f"       -> Flowers with longer petals tend to have wider petals.")
    log(f"     - Sepal length & petal length: r = {corr_matrix.loc['sepal_length', 'petal_length']:.4f} (strong positive)")
    log(f"       -> Larger sepals are associated with larger petals.")
    log(f"     - Sepal width & petal length: r = {corr_matrix.loc['sepal_width', 'petal_length']:.4f} (negative)")
    log(f"       -> This negative correlation is driven by setosa having wide sepals but tiny petals.")

    # 5: Outliers
    log("\n  5. OUTLIERS")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        log(f"     - {col}: {outliers} outlier(s) detected (IQR method)")

    # 6: Classification Potential
    log("\n  6. CLASSIFICATION POTENTIAL")
    log(f"     - Setosa is linearly separable from the other two species.")
    log(f"     - Petal features are the most discriminative for classification.")
    log(f"     - Versicolor and Virginica require more complex decision boundaries.")
    log(f"     - Best feature pair for separation: petal_length + petal_width.")

    # Summary
    log("\n  7. GENERATED VISUALIZATIONS")
    log(f"     01_histograms.png          — Feature distributions by species")
    log(f"     02_boxplots.png            — Box plots for outlier analysis")
    log(f"     03_scatter_plots.png       — Feature relationship scatter plots")
    log(f"     04_pairplot.png            — Full pair plot (all combinations)")
    log(f"     05_violin_plots.png        — Violin plots (distribution shape)")
    log(f"     06_correlation_heatmaps.png — Pearson & Spearman heatmaps")
    log(f"     07_correlation_by_species.png — Per-species correlation matrices")

    log(f"\n{'=' * 70}")
    log("Process Complete: Exploratory Data Analysis (EDA)")
    log(f"{'=' * 70}")



# MAIN EXECUTION

def main():

    log("  TASK 3: EXPLORATORY DATA ANALYSIS (EDA)")

    log("=" * 70 + "\n")

    # Step 0: Load data
    df = load_data()

    # Step 1: Compute summary statistics
    stats_table = compute_summary_statistics(df)

    # Step 2: Create visualizations
    create_visualizations(df)

    # Step 3: Correlation analysis
    corr_matrix = correlation_analysis(df)

    # Step 4: Generate insights report
    generate_insights_report(df, stats_table, corr_matrix)

    # Save full report to file
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Full EDA report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()




