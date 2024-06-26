import itertools
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

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
    """
    Plots a confusion matrix.

    Parameters:
    cm (np.ndarray): Confusion matrix array.
    classes (list): List of class labels.
    normalize (bool): Whether to normalize the matrix. Default is False.
    title (str): Title of the plot. Default is "Confusion matrix".
    cmap (plt.cm): Colormap for the matrix. Default is plt.cm.Blues.

    Returns:
    None: Displays the confusion matrix plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    cm = np.round(cm, 2)
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
    """
    Plots the training and validation accuracy and loss curves of a neural network models
    over epochs.

    Parameters:
    history (dict): A dictionary containing the training history of the models. Expected keys:
                    - "accuracy": Array of training accuracies over epochs.
                    - "val_accuracy": Array of validation accuracies over epochs.
                    - "loss": Array of training losses over epochs.
                    - "val_loss": Array of validation losses over epochs.

    Returns:
    None: Displays the plots using matplotlib.
    """
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Set")
    plt.plot(epochs_range, val_acc, label="Val Set")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Set")
    plt.plot(epochs_range, val_loss, label="Val Set")
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Model Loss")

    plt.tight_layout()
    plt.show()


def plot_precision_recall(
    y_true: Union[list, np.ndarray], y_scores: Union[list, np.ndarray]
) -> None:
    """
    Plots the precision-recall curve based on true labels and predicted scores.

    Parameters:
    y_true (Union[list, np.ndarray]): True binary labels (0 or 1).
    y_scores (Union[list, np.ndarray]): Target scores, can either be probability estimates
                                         of the positive class or confidence values.

    Returns:
    None: Displays the precision-recall curve plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"Precision-Recall curve: AP={average_precision}")
    plt.show()


def plot_roc_curve(
    y_true: Union[list, np.ndarray], y_scores: Union[list, np.ndarray]
) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve for binary classification.

    Parameters:
    y_true (list or np.ndarray): True binary labels (0 or 1).
    y_scores (list or np.ndarray): Predicted probabilities or scores for positive class.

    Returns:
    None: Displays the ROC curve plot using matplotlib.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
