import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

from COV_brain_tumor_detection.config import TEST_DIR, TRAIN_DIR, VAL_DIR


def plot_data_distribution() -> None:
    """
    Plots the distribution of data into training, testing, and validation sets based on image counts.

    Parameters:
    None

    Returns:
    None
    """
    train_yes_count = len(os.listdir(os.path.join(TRAIN_DIR, "yes")))
    train_no_count = len(os.listdir(os.path.join(TRAIN_DIR, "no")))
    test_yes_count = len(os.listdir(os.path.join(TEST_DIR, "yes")))
    test_no_count = len(os.listdir(os.path.join(TEST_DIR, "no")))
    val_yes_count = len(os.listdir(os.path.join(VAL_DIR, "yes")))
    val_no_count = len(os.listdir(os.path.join(VAL_DIR, "no")))

    train_count = train_yes_count + train_no_count
    test_count = test_yes_count + test_no_count
    val_count = val_yes_count + val_no_count

    categories = ["Train", "Test", "Validation"]
    counts = [train_count, test_count, val_count]

    plt.bar(categories, counts, color=["blue", "green", "red"])
    plt.title("Distribution of data into sets")
    plt.xlabel("Data sets")
    plt.ylabel("Number of images")
    plt.show()


def plot_samples(X: np.ndarray, y: np.ndarray, labels_dict: dict, n: int = 5) -> None:
    """
    Plots a grid of sample images from the dataset with corresponding labels.

    Parameters:
    X (np.ndarray): Array of images in numpy format.
    y (np.ndarray): Array of labels corresponding to each image.
    labels_dict (dict): Dictionary mapping label indices to label names.
    n (int): Number of sample images to plot for each label. Default is 5.

    Returns:
    None: Displays the grid of sample images using Matplotlib.
    """
    fig, axes = plt.subplots(2, n, figsize=(20, 8))
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        for i in range(n):
            axes[index, i].imshow(imgs[i][0])
            axes[index, i].axis("off")
            axes[index, i].set_title(labels_dict[index])
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    normalize: bool = False,
    title: str = "Confusion matrix",
    cmap: plt.cm = plt.cm.Blues,
) -> None:
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def plot_model_performance(history: dict) -> None:
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    min_length = min(len(acc), len(val_acc), len(train_loss), len(val_loss))
    acc = acc[:min_length]
    val_acc = val_acc[:min_length]
    train_loss = train_loss[:min_length]
    val_loss = val_loss[:min_length]

    epochs_range = range(1, min_length + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Set")
    plt.plot(epochs_range, val_acc, label="Val Set")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label="Train Set")
    plt.plot(epochs_range, val_loss, label="Val Set")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model Loss")

    plt.tight_layout()
    plt.show()
