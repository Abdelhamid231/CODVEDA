"""
Text Classification using NLP techniques
------------------------------------------
Preprocessing text data, converting to numerical features with TF-IDF,
and classifying sentiment using Naive Bayes and Logistic Regression.

Dataset: Sentiment analysis dataset with social media posts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import string
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
)
from sklearn.preprocessing import LabelEncoder

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "Data Set For Task", "3) Sentiment dataset.csv"
)
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Basic text preprocessing pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                      # remove mentions
    text = re.sub(r"#\w+", "", text)                      # remove hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", "", text)                        # remove numbers
    # remove emojis and special unicode
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()              # normalize whitespace

    # tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def simplify_sentiment(sentiment):
    """Map the many sentiment labels to positive/negative/neutral."""
    sentiment = str(sentiment).strip().lower()

    positive_words = [
        "positive", "joy", "excitement", "happy", "love", "contentment",
        "hope", "gratitude", "enthusiasm", "optimism", "pride", "amusement",
        "relief", "admiration", "thrill", "delight", "elation", "satisfaction",
        "euphoria", "bliss", "serenity", "affection", "inspiration",
        "compassion", "empathy", "cheerfulness", "confidence",
    ]
    negative_words = [
        "negative", "sadness", "anger", "fear", "disgust", "frustration",
        "anxiety", "disappointment", "jealousy", "guilt", "shame", "contempt",
        "boredom", "loneliness", "despair", "grief", "rage", "resentment",
        "melancholy", "hatred", "pessimism", "regret", "embarrassment",
        "helplessness", "panic", "sorrow", "annoyance", "envy",
    ]
    neutral_words = [
        "neutral", "surprise", "curiosity", "indifference", "confusion",
        "ambivalence", "nostalgia", "anticipation",
    ]

    for w in positive_words:
        if w in sentiment:
            return "positive"
    for w in negative_words:
        if w in sentiment:
            return "negative"
    for w in neutral_words:
        if w in sentiment:
            return "neutral"
    return "neutral"


def main():
    # ── Load and explore ───────────────────────────────────────────
    print("Loading sentiment dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # the dataset has many fine-grained sentiments, simplify to 3 classes
    df["text_raw"] = df["Text"].astype(str).str.strip()
    df["sentiment_raw"] = df["Sentiment"].astype(str).str.strip()
    df["label"] = df["sentiment_raw"].apply(simplify_sentiment)

    # drop any rows with empty text
    df = df[df["text_raw"].str.len() > 3].reset_index(drop=True)

    print(f"\n  Simplified label distribution:")
    label_counts = df["label"].value_counts()
    for lab, cnt in label_counts.items():
        print(f"    {lab:10s}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # ── Text preprocessing ─────────────────────────────────────────
    print("\nPreprocessing text...")
    df["text_clean"] = df["text_raw"].apply(clean_text)

    # drop empty after cleaning
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    print(f"  Samples after cleaning: {len(df)}")

    # show a few examples
    print("\n  Sample texts processing completed successfully.")

    # ── Visualize class distribution ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}
    label_counts.plot(kind="bar", ax=ax, color=[colors.get(l, "#95a5a6") for l in label_counts.index])
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png"))
    plt.close()
    print("  Saved class distribution plot")

    # ── TF-IDF Vectorization ──────────────────────────────────────
    print("\nConverting text to TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(df["text_clean"])

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    class_names = le.classes_

    print(f"  TF-IDF matrix: {X.shape}")
    print(f"  Classes: {list(class_names)}")

    # top TF-IDF terms
    feature_names = tfidf.get_feature_names_out()
    mean_tfidf = X.mean(axis=0).A1
    top_idx = mean_tfidf.argsort()[-15:][::-1]
    print(f"\n  Top 15 TF-IDF terms:")
    for idx in top_idx:
        print(f"    {feature_names[idx]:20s} (score: {mean_tfidf[idx]:.4f})")

    # ── Train/test split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ── Train and compare models ──────────────────────────────────
    print("\nTraining classifiers...\n")
    models = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    best_model = None
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "CV_Mean": cv.mean(), "CV_Std": cv.std(),
        })

        if acc > best_acc:
            best_acc = acc
            best_model = (name, model, y_pred)

        print(f"  {name:22s}  Acc={acc:.4f}  F1={f1:.4f}  CV={cv.mean():.4f}")

    # ── Detailed results for best model ───────────────────────────
    best_name, best_clf, best_preds = best_model
    print(f"\n  Best model: {best_name}")
    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(y_test, best_preds, target_names=class_names))

    # confusion matrix
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, linewidths=1,
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_confusion_matrix.png"))
    plt.close()

    # model comparison chart
    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.2
    ax.bar(x - width, results_df["Accuracy"], width, label="Accuracy", color="#3498db")
    ax.bar(x, results_df["Precision"], width, label="Precision", color="#2ecc71")
    ax.bar(x + width, results_df["F1"], width, label="F1-Score", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_model_comparison.png"))
    plt.close()

    print("  Saved confusion matrix and comparison plots")

    # ── Summary ────────────────────────────────────────────────────

    print("NLP TEXT CLASSIFICATION SUMMARY")

    print(f"  Classes: {list(class_names)}")
    print(f"  Vectorizer: TF-IDF (max_features=5000, bigrams)")
    print(f"  Best model: {best_name} (Acc={best_acc:.4f})")
    print(f"\n  All results:")
    print(results_df[["Model", "Accuracy", "F1", "CV_Mean"]].to_string(index=False))

    # save report
    with open(os.path.join(SCRIPT_DIR, "nlp_report.txt"), "w") as f:
        f.write("NLP Text Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: Sentiment dataset ({len(df)} samples)\n")
        f.write(f"Labels: {list(class_names)}\n\n")
        f.write("Results:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write(f"Best model: {best_name}\n")
        f.write(classification_report(y_test, best_preds, target_names=class_names))

    print("\nDone!")


if __name__ == "__main__":
    main()




