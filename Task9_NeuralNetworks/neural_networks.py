"""
Neural Network Classification with TensorFlow/Keras
-----------------------------------------------------
Building a feed-forward neural network to classify the Iris dataset
and the MNIST handwritten digits dataset.

Covers network design, training with backpropagation, evaluation,
and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# suppress TF info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def train_iris_nn():
    """Part 1: Neural network on the Iris structured dataset."""
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix

    print("PART 1: Neural Network on Iris Dataset")

    # load iris
    DATA_PATH = os.path.join(
        os.path.dirname(SCRIPT_DIR), "Data Set For Task", "1) iris.csv"
    )
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["species"])
    class_names = le.classes_

    # scale and split
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # one-hot encode targets for the network
    y_train_oh = keras.utils.to_categorical(y_train, 3)
    y_test_oh = keras.utils.to_categorical(y_test, 3)

    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}, Classes: {len(class_names)}")

    # build the network
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n  Model architecture:")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    # train
    print("\n  Training...")
    history = model.fit(
        X_train, y_train_oh,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        verbose=0,
    )

    # evaluate
    loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    print(f"\n  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Iris NN - Accuracy Curves")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Iris NN - Loss Curves")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_iris_training_curves.png"))
    plt.close()
    print("  Saved training curves")

    return acc


def train_mnist_nn():
    """Part 2: Neural network on MNIST digits."""
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    print("PART 2: Neural Network on MNIST Digits")

    # load MNIST
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Labels: 0-9 (10 digit classes)")

    # normalize pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # flatten 28x28 images to vectors
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat = X_test.reshape(-1, 784)

    # one-hot encode
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)

    # visualize some samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X_train[i], cmap="gray")
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis("off")
    plt.suptitle("MNIST Sample Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_mnist_samples.png"))
    plt.close()
    print("  Saved sample images")

    # build the network
    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n  Model architecture:")
    model.summary(print_fn=lambda x: print(f"    {x}"))

    # train
    print("\n  Training (this may take a minute)...")
    history = model.fit(
        X_train_flat, y_train_oh,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        verbose=0,
    )

    # evaluate
    loss, acc = model.evaluate(X_test_flat, y_test_oh, verbose=0)
    y_pred = model.predict(X_test_flat, verbose=0).argmax(axis=1)

    print(f"\n  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")

    # training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("MNIST NN - Accuracy Curves")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("MNIST NN - Loss Curves")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_mnist_training_curves.png"))
    plt.close()
    print("  Saved training curves")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=range(10), yticklabels=range(10),
                annot_kws={"fontsize": 10})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"MNIST Confusion Matrix (Accuracy: {acc:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_mnist_confusion_matrix.png"))
    plt.close()
    print("  Saved confusion matrix")

    # ── Hyperparameter tuning comparison ───────────────────────────
    print("\n  Testing different hyperparameters...")
    configs = [
        {"lr": 0.01, "batch": 256, "name": "lr=0.01, batch=256"},
        {"lr": 0.001, "batch": 128, "name": "lr=0.001, batch=128 (baseline)"},
        {"lr": 0.0005, "batch": 64, "name": "lr=0.0005, batch=64"},
    ]

    tuning_results = []
    for cfg in configs:
        m = keras.Sequential([
            keras.layers.Input(shape=(784,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        h = m.fit(X_train_flat, y_train_oh, validation_split=0.1,
                  epochs=10, batch_size=cfg["batch"], verbose=0)
        _, test_acc = m.evaluate(X_test_flat, y_test_oh, verbose=0)
        tuning_results.append({
            "Config": cfg["name"],
            "Test Acc": test_acc,
            "Val Acc": h.history["val_accuracy"][-1],
        })
        print(f"    {cfg['name']:35s} -> Test Acc: {test_acc:.4f}")

    # plot tuning results
    fig, ax = plt.subplots(figsize=(10, 5))
    tune_df = pd.DataFrame(tuning_results)
    x = np.arange(len(tune_df))
    ax.bar(x, tune_df["Test Acc"], color=["#e74c3c", "#2ecc71", "#3498db"],
           edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tune_df["Config"], rotation=15, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Hyperparameter Tuning Results")
    ax.set_ylim(0.95, 1.0)
    for i, v in enumerate(tune_df["Test Acc"]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_hyperparameter_tuning.png"))
    plt.close()
    print("  Saved tuning results plot")

    return acc


def main():
    print("\nNeural Network Classification")
    print("Using TensorFlow/Keras\n")

    iris_acc = train_iris_nn()
    mnist_acc = train_mnist_nn()

    # final summary

    print("NEURAL NETWORK RESULTS SUMMARY")

    print(f"    Architecture: 4 -> 64 -> 32 -> 3")
    print(f"    Test Accuracy: {iris_acc:.4f}")

    print(f"    Architecture: 784 -> 256 -> 128 -> 64 -> 10")
    print(f"    Test Accuracy: {mnist_acc:.4f}")
    print(f"\n  Both models trained with Adam optimizer")
    print(f"  and categorical cross-entropy loss.")

    # save report
    with open(os.path.join(SCRIPT_DIR, "neural_network_report.txt"), "w") as f:
        f.write("Neural Network Classification Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Iris NN: Accuracy = {iris_acc:.4f}\n")
        f.write(f"  Architecture: Input(4) -> Dense(64) -> Dense(32) -> Dense(3)\n")
        f.write(f"  Optimizer: Adam (lr=0.001)\n\n")
        f.write(f"MNIST NN: Accuracy = {mnist_acc:.4f}\n")
        f.write(f"  Architecture: Input(784) -> Dense(256) -> Dense(128) -> Dense(64) -> Dense(10)\n")
        f.write(f"  Optimizer: Adam (lr=0.001)\n")

    print("\nDone!")


if __name__ == "__main__":
    main()




