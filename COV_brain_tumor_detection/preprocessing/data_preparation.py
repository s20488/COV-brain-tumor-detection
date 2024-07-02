import os
import shutil

import cv2
import imutils
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
    """
    Splits dataset into training, testing, and validation sets and copies files to corresponding directories.

    Parameters:
        img_path (str): Path to the image dataset directory.
        train_size (float): Proportion of the dataset for training. Default is 80%.
        test_size (float): Proportion of the dataset for testing. Default is 10%.
        val_size (float): Proportion of the dataset for validation. Default is 10%.

    Returns:
        None: Creates directories and copies files.
    """
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
    """
    Loads image data from a directory and prepares it for machine learning tasks.

    Parameters:
        dir_path (str): The path to the directory containing subdirectories of image data.
        img_size (tuple): The desired size of the images after resizing. Default is IMG_SIZE.

    Returns:
        tuple: A tuple containing:
            - X (np.array): An array of resized images.
            - y (np.array): An array of corresponding labels.
            - labels (dict): A dictionary mapping label indices to category names.
    """
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


def preprocess_images(
    image_set: np.ndarray, padding: int = 0, output_size: tuple = IMG_SIZE
) -> np.ndarray:
    """
    Preprocesses images for machine learning tasks.

    Parameters:
        image_set (np.ndarray): An array of images to preprocess.
        padding (int): The amount of padding around the detected region of interest. Default is 0.
        output_size (tuple): The desired size of the output images after preprocessing. Default is IMG_SIZE.

    Returns:
        np.ndarray: An array of preprocessed images.
    """
    cropped_images = []

    for image in image_set:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        _, binary_thresh = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)

        binary_thresh = cv2.erode(binary_thresh, None, iterations=2)
        binary_thresh = cv2.dilate(binary_thresh, None, iterations=2)

        contours = cv2.findContours(
            binary_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)
        largest_contour = max(contours, key=cv2.contourArea)

        leftmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        rightmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
        topmost_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        bottommost_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

        top = max(topmost_point[1] - padding, 0)
        bottom = min(bottommost_point[1] + padding, image.shape[0])
        left = max(leftmost_point[0] - padding, 0)
        right = min(rightmost_point[0] + padding, image.shape[1])

        cropped_image = image[top:bottom, left:right].copy()
        cropped_image = cv2.resize(cropped_image, output_size)
        cropped_images.append(cropped_image)

    return np.array(cropped_images)
