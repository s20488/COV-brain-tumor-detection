import cv2
import imutils
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from COV_brain_tumor_detection.config import IMG_SIZE, SEED, TRAIN_DIR, VAL_DIR


def preprocess_images(
    image_set: np.ndarray, padding: int = 0, output_size: tuple = IMG_SIZE
) -> np.ndarray:
    """
    Preprocesses images by cropping around the largest contour,
    applying Gaussian blur, thresholding, erosion, dilation, and resizing.

    Parameters:
    image_set (np.ndarray): Input images as a NumPy array.
    padding (int): Padding around detected contour (default: 0).
    output_size (tuple): Desired output size after resizing (default: IMG_SIZE).

    Returns:
    np.ndarray: Preprocessed images as a NumPy array.
    """
    cropped_images = []

    for image in image_set:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply a threshold value for image binarisation
        _, binary_thresh = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)

        binary_thresh = cv2.erode(binary_thresh, None, iterations=2)
        binary_thresh = cv2.dilate(binary_thresh, None, iterations=2)

        contours = cv2.findContours(
            binary_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)
        largest_contour = max(contours, key=cv2.contourArea)

        # Define the outermost points of the largest contour
        leftmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        rightmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
        topmost_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        bottommost_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

        # Define borders for cropping the image by adding padding
        top = max(topmost_point[1] - padding, 0)
        bottom = min(bottommost_point[1] + padding, image.shape[0])
        left = max(leftmost_point[0] - padding, 0)
        right = min(rightmost_point[0] + padding, image.shape[1])

        cropped_image = image[top:bottom, left:right].copy()

        cropped_image = cv2.resize(cropped_image, output_size)
        cropped_images.append(preprocess_input(cropped_image))

    return np.array(cropped_images)


def create_data_generators() -> tuple:
    """
    Creates and returns train and validation data generators for image data.

    Returns:
    tuple: A tuple containing train and validation data generators.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
    )

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        color_mode="rgb",
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="binary",
        seed=SEED,
    )

    validation_generator = validation_datagen.flow_from_directory(
        VAL_DIR,
        color_mode="rgb",
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode="binary",
        seed=SEED,
    )

    return train_generator, validation_generator
