import itertools
import os
import re
import shutil

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (224, 224)
random_seed = 3

directories_to_remove = ["train/", "test/", "val/"]

for directory in directories_to_remove:
    if os.path.exists(directory):
        shutil.rmtree(directory)


def split_data(
    img_path: str,
    train_size: float = 0.7,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> None:
    directories = ["train/yes", "train/no", "test/yes", "test/no", "val/yes", "val/no"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    labels = [label for label in os.listdir(img_path) if not label.startswith(".")]

    for label in labels:
        files = os.listdir(os.path.join(img_path, label))
        train_files, test_files = train_test_split(
            files,
            train_size=train_size + val_size,
            test_size=test_size,
            random_state=random_seed,
        )
        train_files, val_files = train_test_split(
            train_files,
            train_size=train_size / (train_size + val_size),
            random_state=random_seed,
        )

        for file_name in train_files:
            shutil.copy(
                os.path.join(img_path, label, file_name),
                os.path.join("train", label.lower(), file_name),
            )
        for file_name in test_files:
            shutil.copy(
                os.path.join(img_path, label, file_name),
                os.path.join("test", label.lower(), file_name),
            )
        for file_name in val_files:
            shutil.copy(
                os.path.join(img_path, label, file_name),
                os.path.join("val", label.lower(), file_name),
            )


split_data("brain_tumor_dataset")


def plot_data_distribution() -> None:
    train_yes_count = len(os.listdir(os.path.join("train", "yes")))
    train_no_count = len(os.listdir(os.path.join("train", "no")))
    test_yes_count = len(os.listdir(os.path.join("test", "yes")))
    test_no_count = len(os.listdir(os.path.join("test", "no")))
    val_yes_count = len(os.listdir(os.path.join("val", "yes")))
    val_no_count = len(os.listdir(os.path.join("val", "no")))

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


plot_data_distribution()


def plot_samples(X: np.ndarray, y: np.ndarray, labels_dict: dict, n: int = 5) -> None:
    fig, axes = plt.subplots(2, n, figsize=(20, 8))
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        for i in range(n):
            axes[index, i].imshow(imgs[i][0])
            axes[index, i].axis("off")
            axes[index, i].set_title(labels_dict[index])
    plt.show()


def load_data(dir_path: str, img_size: tuple = img_size) -> tuple:
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


X_train, y_train, labels = load_data("train/")
X_test, y_test, _ = load_data("test/")
X_val, y_val, _ = load_data("val/")

plot_samples(X_train, y_train, labels)


def crop_images(
    image_set: np.ndarray, padding: int = 0, output_size: tuple = img_size
) -> np.ndarray:
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


X_train_crop = crop_images(X_train)
X_val_crop = crop_images(X_val)
X_test_crop = crop_images(X_test)

plot_samples(X_train_crop, y_train, labels)


def save_new_images(x_set: np.ndarray, y_set: np.ndarray, folder_name: str) -> None:
    for index, (img, imclass) in enumerate(zip(x_set, y_set)):
        class_folder = "yes/" if imclass == 1 else "no/"
        filename = f"{index}.jpg"
        cv2.imwrite(f"{folder_name}{class_folder}{filename}", img)


save_new_images(X_train_crop, y_train, folder_name="train/")
save_new_images(X_val_crop, y_val, folder_name="val/")
save_new_images(X_test_crop, y_test, folder_name="test/")


def clean_directory(folder_name: str) -> None:
    pattern = re.compile(r"^\d+\.jpg$")
    for filename in os.listdir(folder_name):
        if not pattern.match(filename):
            os.remove(os.path.join(folder_name, filename))


clean_directory("train/yes/")
clean_directory("train/no/")
clean_directory("val/yes/")
clean_directory("val/no/")
clean_directory("test/yes/")
clean_directory("test/no/")


def preprocess_imgs(set_name: np.ndarray, img_size: tuple = img_size) -> np.ndarray:
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)


X_train_prep = preprocess_imgs(set_name=X_train_crop)
X_test_prep = preprocess_imgs(set_name=X_test_crop)
X_val_prep = preprocess_imgs(set_name=X_val_crop)

train_dir = "train/"
val_dir = "val/"

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

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode="rgb",
    target_size=img_size,
    batch_size=32,
    class_mode="binary",
    seed=random_seed,
)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    color_mode="rgb",
    target_size=img_size,
    batch_size=16,
    class_mode="binary",
    seed=random_seed,
)

train_steps = len(train_generator)
val_steps = len(validation_generator)

vgg16_weight_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

base_model = VGG16(
    weights=vgg16_weight_path, include_top=False, input_shape=img_size + (3,)
)

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation="sigmoid"))

model.layers[0].trainable = False

model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=["accuracy"],
)

model.summary()

EPOCHS = 30
es = EarlyStopping(monitor="val_accuracy", mode="max", patience=6)

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=[es],
)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    normalize: bool = False,
    title: str = "Confusion matrix",
    cmap: plt.cm = plt.cm.Blues,
) -> None:
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


# plot model performance
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(history.epoch) + 1)

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

predictions = model.predict(X_test_prep)
predictions = [1 if x > 0.5 else 0 for x in predictions]

accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy = %.2f" % accuracy)

confusion_mtx = confusion_matrix(y_test, predictions)
plot_confusion_matrix(confusion_mtx, classes=list(labels.items()), normalize=False)
