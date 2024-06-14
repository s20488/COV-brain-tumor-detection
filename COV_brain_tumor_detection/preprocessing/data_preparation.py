import os
import shutil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from COV_brain_tumor_detection.config import (
    IMG_SIZE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


def split_data(
    img_path: str,
    train_size: float = 0.8,
    test_size: float = 0.1,
    val_size: float = 0.1,
) -> None:
    directories = [
        TRAIN_DIR + "yes",
        TRAIN_DIR + "no",
        TEST_DIR + "yes",
        TEST_DIR + "no",
        VAL_DIR + "yes",
        VAL_DIR + "no",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    labels = [label for label in os.listdir(img_path) if not label.startswith(".")]

    for label in labels:
        files = os.listdir(os.path.join(img_path, label))
        train_files, test_files = train_test_split(
            files,
            train_size=train_size + val_size,
            test_size=test_size,
            random_state=SEED,
        )
        train_files, val_files = train_test_split(
            train_files,
            train_size=train_size / (train_size + val_size),
            random_state=SEED,
        )

        for file_name in train_files:
            shutil.copy(
                os.path.join(img_path, label, file_name),
                os.path.join(
                    "brain_tumor_dataset_evolved/train", label.lower(), file_name
                ),
            )
        for file_name in test_files:
            shutil.copy(
                os.path.join(img_path, label, file_name),
                os.path.join(
                    "brain_tumor_dataset_evolved/test", label.lower(), file_name
                ),
            )
        for file_name in val_files:
            shutil.copy(
                os.path.join(img_path, label, file_name),
                os.path.join(
                    "brain_tumor_dataset_evolved/val", label.lower(), file_name
                ),
            )


def load_data(dir_path: str, img_size: tuple = IMG_SIZE) -> tuple:
    X = []
    y = []
    i = 0
    labels = dict()
    for path in sorted(os.listdir(dir_path)):
        if not path.startswith("."):
            labels[i] = path
            for file in os.listdir(os.path.join(dir_path, path)):
                if not file.startswith("."):
                    img = cv2.imread(os.path.join(dir_path, path, file))
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        X.append(img)
                        y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    return X, y, labels
