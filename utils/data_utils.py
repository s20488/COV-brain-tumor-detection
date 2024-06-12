import os
import re
import shutil

import cv2
import numpy as np


def remove_directories(directories: list[str]) -> None:
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)


def save_new_images(x_set: np.ndarray, y_set: np.ndarray, folder_name: str) -> None:
    for index, (img, imclass) in enumerate(zip(x_set, y_set)):
        class_folder = "yes/" if imclass == 1 else "no/"
        filename = f"{index}.jpg"
        cv2.imwrite(f"{folder_name}{class_folder}{filename}", img)


def clean_directory(folder_name: str) -> None:
    pattern = re.compile(r"^\d+\.jpg$")
    for root, dirs, files in os.walk(folder_name):
        for filename in files:
            if not pattern.match(filename):
                os.remove(os.path.join(root, filename))
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            if not os.listdir(dir_path):
                shutil.rmtree(dir_path)
