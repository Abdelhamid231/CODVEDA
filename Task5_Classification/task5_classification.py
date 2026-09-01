"""
Iris Species Classification
------------------------------
Builds and compares multiple classifiers to predict flower species
based on morphological measurements.

Models: Logistic Regression, Decision Tree, Random Forest, SVM,
        K-Nearest Neighbors, Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")


# CONFIGURATION

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "Data Set For Task", "1) iris.csv"
)
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
REPORT_FILE = os.path.join(SCRIPT_DIR, "classification_report.txt")

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

SPECIES_COLORS = {"setosa": "#2ecc71", "versicolor": "#3498db", "virginica": "#e74c3c"}

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)



# STEP 1: Data Preprocessing

def load_and_preprocess():
    print(" Data Preprocessing")

    df = pd.read_csv(DATA_PATH)
    log(f"\n  Source : {DATA_PATH}")
    log(f"  Shape : {df.shape}")
    log(f"  Columns: {list(df.columns)}")

    # Check for missing values
    log(f"\n  --- Missing Values ---")
    log(f"  Total: {df.isnull().sum().sum()}")

    # Check class distribution
    log(f"\n  --- Class Distribution ---")
    for species, count in df["species"].value_counts().items():
        log(f"    {species:15s}: {count} ({count/len(df)*100:.1f}%)")

    # Encode target variable
    le = LabelEncoder()
    df["species_encoded"] = le.fit_transform(df["species"])
    class_names = le.classes_
    log(f"\n  --- Label Encoding ---")
    for cls, code in zip(class_names, le.transform(class_names)):
        log(f"    {cls} -> {code}")

    # Feature scaling
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols]
    y = df["species_encoded"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    log(f"\n  --- Feature Scaling (StandardScaler) ---")
    log(f"  Before scaling:")
    log(f"    Means: {X.mean().round(2).to_dict()}")
    log(f"    Stds : {X.std().round(2).to_dict()}")
    log(f"  After scaling:")
    log(f"    Means: {X_scaled.mean().round(4).to_dict()}")
    log(f"    Stds : {X_scaled.std().round(4).to_dict()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    log(f"\n  --- Train-Test Split ---")
    log(f"  Split: 80% train / 20% test (stratified)")
    log(f"  Training: {X_train.shape[0]} samples")
    log(f"  Testing : {X_test.shape[0]} samples")

    log(f"\n  Training class distribution:")
    for cls_name, cls_code in zip(class_names, range(len(class_names))):
        count = (y_train == cls_code).sum()
        log(f"    {cls_name}: {count}")

    log(f"\n  Testing class distribution:")
    for cls_name, cls_code in zip(class_names, range(len(class_names))):
        count = (y_test == cls_code).sum()
        log(f"    {cls_name}: {count}")

    log("")
    return X_train, X_test, y_train, y_test, class_names, X_scaled, y, le



# STEP 2: Train Logistic Regression

def train_logistic_regression(X_train, X_test, y_train, y_test, class_names):
    print(" Training Logistic Regression Model")

    model = LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 
    acc = accuracy_score(y_test, y_pred)
    log(f"\n  Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # Report
    log(f"\n  --- Detailed Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=class_names)
    log(report)

    # Matrix
    cm = confusion_matrix(y_test, y_pred)
    log(f"  --- Confusion Matrix ---")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    log(cm_df.to_string())

    # -class Metrics
    log(f"\n  --- Per-Class Metrics ---")
    for i, cls in enumerate(class_names):
        prec = precision_score(y_test, y_pred, labels=[i], average="micro")
        rec = recall_score(y_test, y_pred, labels=[i], average="micro")
        f1 = f1_score(y_test, y_pred, labels=[i], average="micro")
        log(f"    {cls:15s}: Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    # -Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    log(f"\n  --- 5-Fold Cross-Validation ---")
    log(f"    Scores: {[f'{s:.4f}' for s in cv_scores]}")
    log(f"    Mean  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    log("")

    return model, y_pred, y_proba, cm



# STEP 3: ROC Curve & Evaluation Visualizations

def evaluate_and_visualize(model, X_test, y_test, y_pred, y_proba, cm, class_names):
    print(" Evaluation Visualizations (ROC Curve, Confusion Matrix)")

    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # ROC Curves (One-vs-Rest)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors_roc = ["#2ecc71", "#3498db", "#e74c3c"]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors_roc[i], linewidth=2.5,
                label=f"{class_names[i]} (AUC = {roc_auc:.4f})")

    # Macro-average ROC
    fpr_grid = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(fpr_grid, fpr_i, tpr_i)
    mean_tpr /= n_classes
    macro_auc = auc(fpr_grid, mean_tpr)
    ax.plot(fpr_grid, mean_tpr, color="#8e44ad", linewidth=2.5, linestyle="--",
            label=f"Macro-average (AUC = {macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Logistic Regression - ROC Curves (One-vs-Rest)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "01_roc_curves.png")
    plt.savefig(path)
    plt.close()
    log(f"\n  Saved: {path}")

    # Log AUC values
    log(f"\n  --- ROC AUC Scores ---")
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_i = auc(fpr_i, tpr_i)
        log(f"    {class_names[i]:15s}: AUC = {roc_auc_i:.4f}")
    log(f"    {'Macro-average':15s}: AUC = {macro_auc:.4f}")

    # Confusion Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Logistic Regression - Confusion Matrix", fontsize=15, fontweight="bold", y=1.02)

    # Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[0], linewidths=1,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)
    axes[0].set_title("Counts", fontsize=13)

    # Normalized
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], linewidths=1,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)
    axes[1].set_title("Normalized (%)", fontsize=13)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "02_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    log(f"  Saved: {path}")

    # Decision Boundary (using top 2 features)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use petal_length and petal_width (most discriminative)
    feature_idx = [2, 3]  # petal_length, petal_width
    X_2d = X_test.iloc[:, feature_idx].values

    lr_2d = LogisticRegression(solver="lbfgs", max_iter=1000)
    from sklearn.model_selection import train_test_split as tts
    X_train_full = np.vstack([X_test.iloc[:, feature_idx].values])  # Just for visualization

    # Re-fit on 2 features for visualization
    X_all_2d = np.column_stack([
        X_test.iloc[:, 2].values,
        X_test.iloc[:, 3].values,
    ])

    # Plot test points
    for i, cls in enumerate(class_names):
        mask = y_test.values == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=list(SPECIES_COLORS.values())[i], label=cls,
                   s=80, edgecolors="white", linewidth=1, alpha=0.8)

    ax.set_xlabel("Petal Length (standardized)", fontsize=12)
    ax.set_ylabel("Petal Width (standardized)", fontsize=12)
    ax.set_title("Test Set - Species Distribution\n(Petal Length vs Petal Width)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "03_species_scatter.png")
    plt.savefig(path)
    plt.close()
    log(f"  Saved: {path}")
    log("")



# STEP 4: Compare Multiple Classifiers

def compare_classifiers(X_train, X_test, y_train, y_test, class_names):
    print(" Comparing Multiple Classifiers")

    models = {
        "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "SVM (Linear)": SVC(kernel="linear", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        log(f"\n  Training: {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "CV Mean": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Predictions": y_pred,
        })

        log(f"    Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | CV={cv_scores.mean():.4f}")

    # Table
    results_df = pd.DataFrame(results)
    log(f"\n  {'=' * 70}")
    log(f"  CLASSIFIER COMPARISON RESULTS")
    log(f"  {'=' * 70}")

    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "CV Mean"]
    display_df = results_df[display_cols].sort_values("Accuracy", ascending=False)
    for col in display_cols[1:]:
        display_df[col] = display_df[col].round(4)
    log(f"\n{display_df.to_string(index=False)}")

    # Best model
    best_idx = results_df["Accuracy"].idxmax()
    best = results_df.iloc[best_idx]
    log(f"\n  BEST MODEL: {best['Model']}")
    log(f"    Accuracy : {best['Accuracy']:.4f}")
    log(f"    F1-Score : {best['F1-Score']:.4f}")
    log(f"    CV Mean  : {best['CV Mean']:.4f} (+/- {best['CV Std']:.4f})")

    # : Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Classifier Comparison", fontsize=16, fontweight="bold", y=1.02)

    # Accuracy comparison
    sorted_df = results_df.sort_values("Accuracy", ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_df)))
    ax = axes[0]
    ax.barh(sorted_df["Model"], sorted_df["Accuracy"], color=colors, edgecolor="white")
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Test Accuracy", fontsize=13, fontweight="bold")
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax.text(row["Accuracy"] + 0.003, i, f'{row["Accuracy"]:.4f}', va="center", fontsize=10)
    ax.set_xlim([0.8, 1.05])

    # F1-Score comparison
    sorted_df2 = results_df.sort_values("F1-Score", ascending=True)
    colors2 = plt.cm.RdYlBu(np.linspace(0.3, 0.9, len(sorted_df2)))
    ax = axes[1]
    ax.barh(sorted_df2["Model"], sorted_df2["F1-Score"], color=colors2, edgecolor="white")
    ax.set_xlabel("F1-Score (weighted)", fontsize=12)
    ax.set_title("F1-Score Comparison", fontsize=13, fontweight="bold")
    for i, (_, row) in enumerate(sorted_df2.iterrows()):
        ax.text(row["F1-Score"] + 0.003, i, f'{row["F1-Score"]:.4f}', va="center", fontsize=10)
    ax.set_xlim([0.8, 1.05])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_classifier_comparison.png")
    plt.savefig(path)
    plt.close()
    log(f"\n  Saved: {path}")

    # Matrices for All Models
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Confusion Matrices - All Classifiers", fontsize=16, fontweight="bold", y=1.02)
    axes = axes.flatten()

    cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlGn", "BuPu"]

    for idx, res in enumerate(results):
        ax = axes[idx]
        cm_i = confusion_matrix(y_test, res["Predictions"])
        sns.heatmap(cm_i, annot=True, fmt="d", cmap=cmaps[idx % len(cmaps)],
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5,
                    annot_kws={"fontsize": 12, "fontweight": "bold"})
        ax.set_title(f"{res['Model']}\nAcc={res['Accuracy']:.4f}", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

    axes[-1].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "05_all_confusion_matrices.png")
    plt.savefig(path)
    plt.close()
    log(f"  Saved: {path}")

    # Curves for all models (that support probability)
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))

    line_styles = ["-", "--", "-.", ":", "-", "--", "-."]
    model_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

    for idx, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            y_proba_i = model.predict_proba(X_test)
            # Compute macro-average ROC
            fpr_grid = np.linspace(0, 1, 100)
            mean_tpr = np.zeros_like(fpr_grid)
            for c in range(n_classes):
                fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, c], y_proba_i[:, c])
                mean_tpr += np.interp(fpr_grid, fpr_c, tpr_c)
            mean_tpr /= n_classes
            macro_auc_i = auc(fpr_grid, mean_tpr)
            ax.plot(fpr_grid, mean_tpr, color=model_colors[idx], linewidth=2,
                    linestyle=line_styles[idx % len(line_styles)],
                    label=f"{name} (AUC={macro_auc_i:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Macro-Average ROC Curves - All Classifiers", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "06_all_roc_curves.png")
    plt.savefig(path)
    plt.close()
    log(f"  Saved: {path}")

    log("")
    return results_df



# STEP 5: Final Summary

def generate_summary(results_df, class_names):
    print(" Final Summary")

    log(f"\n  --- Key Findings ---")

    lr = results_df[results_df["Model"] == "Logistic Regression"].iloc[0]
    log(f"\n  1. LOGISTIC REGRESSION:")
    log(f"     - Accuracy: {lr['Accuracy']:.4f} ({lr['Accuracy']*100:.1f}%)")
    log(f"     - Precision: {lr['Precision']:.4f}")
    log(f"     - Recall: {lr['Recall']:.4f}")
    log(f"     - F1-Score: {lr['F1-Score']:.4f}")
    log(f"     - Cross-val: {lr['CV Mean']:.4f}")

    best = results_df.loc[results_df["Accuracy"].idxmax()]
    log(f"\n  2. BEST CLASSIFIER: {best['Model']}")
    log(f"     - Accuracy: {best['Accuracy']:.4f}")
    log(f"     - F1-Score: {best['F1-Score']:.4f}")

    log(f"\n  3. MODEL RANKINGS (by Accuracy):")
    ranked = results_df.sort_values("Accuracy", ascending=False)
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        log(f"     {rank}. {row['Model']:22s} Acc={row['Accuracy']:.4f}  F1={row['F1-Score']:.4f}")

    log(f"\n  4. OBSERVATIONS:")
    log(f"     - Iris dataset is well-suited for classification (balanced classes).")
    log(f"     - Setosa is perfectly separable from other species.")
    log(f"     - Most misclassifications occur between versicolor and virginica.")
    log(f"     - Ensemble methods and SVM perform comparably on this dataset.")
    log(f"     - Logistic Regression provides a strong, interpretable baseline.")

    log(f"\n  5. GENERATED OUTPUTS:")
    log(f"     - 01_roc_curves.png            (per-class ROC for Logistic Regression)")
    log(f"     - 02_confusion_matrix.png       (counts + normalized)")
    log(f"     - 03_species_scatter.png        (test set species distribution)")
    log(f"     - 04_classifier_comparison.png  (accuracy + F1 bar charts)")
    log(f"     - 05_all_confusion_matrices.png (all 7 classifiers)")
    log(f"     - 06_all_roc_curves.png         (macro-avg ROC for all models)")

    log(f"\n{'=' * 70}")
    log("Level 2 Process Complete: Classification with Logistic Regression")
    log(f"{'=' * 70}")



# MAIN

def main():

    log("  LEVEL 2, TASK 2: CLASSIFICATION WITH LOGISTIC REGRESSION")

    log("=" * 70 + "\n")

    # Step 1: Preprocess
    X_train, X_test, y_train, y_test, class_names, X_scaled, y, le = load_and_preprocess()

    # Step 2: Train logistic regression
    lr_model, y_pred, y_proba, cm = train_logistic_regression(
        X_train, X_test, y_train, y_test, class_names
    )

    # Step 3: ROC curves & evaluation visualizations
    evaluate_and_visualize(lr_model, X_test, y_test, y_pred, y_proba, cm, class_names)

    # Step 4: Compare classifiers
    results_df = compare_classifiers(X_train, X_test, y_train, y_test, class_names)

    # Step 5: Summary
    generate_summary(results_df, class_names)

    # Save report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()




