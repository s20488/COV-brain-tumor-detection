import os
import re
import shutil

import cv2
import numpy as np


def save_new_images(x_set: np.ndarray, y_set: np.ndarray, folder_name: str) -> None:
    """
    Saves images from a numpy array `x_set` into specified folders based on labels `y_set`.

    Parameters:
        x_set (np.ndarray): An array of images to be saved.
        y_set (np.ndarray): An array of labels corresponding to each image (1 for 'yes', 0 for 'no').
        folder_name (str): The path to the main folder where images will be saved.

    Returns:
        None. This function saves images to specified folders.
    """
    for index, (img, imclass) in enumerate(zip(x_set, y_set)):
        class_folder = "yes/" if imclass == 1 else "no/"
        filename = f"{index}.jpg"
        cv2.imwrite(f"{folder_name}{class_folder}{filename}", img)


def clean_directory(folder_name: str) -> None:
    """
    Cleans up a directory by removing files that do not match a specific pattern and removes empty directories.

    Parameters:
        folder_name (str): The name of the folder to clean up.

    Returns:
        None. This function modifies the directory by deleting files and directories.
    """
    pattern = re.compile(r"^\d+\.jpg$")
    for root, dirs, files in os.walk(folder_name):
        for filename in files:
            if not pattern.match(filename):
                os.remove(os.path.join(root, filename))
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            if not os.listdir(dir_path):
                shutil.rmtree(dir_path)
