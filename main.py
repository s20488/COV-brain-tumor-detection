from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix

from config import TEST_DIR, TRAIN_DIR, VAL_DIR
from data_preprocessing.data_processing import load_data, split_data
from data_preprocessing.image_augmentation import (
    create_data_generators,
    preprocess_images,
)
from model.model import create_model
from utils.data_utils import clean_directory, remove_directories, save_new_images
from visualization.plot_results import (
    plot_confusion_matrix,
    plot_data_distribution,
    plot_model_performance,
    plot_samples,
)

remove_directories([TRAIN_DIR, TEST_DIR, VAL_DIR])

split_data("brain_tumor_dataset")

plot_data_distribution()

X_train, y_train, labels = load_data(TRAIN_DIR)
X_test, y_test, _ = load_data(TEST_DIR)
X_val, y_val, _ = load_data(VAL_DIR)

plot_samples(X_train, y_train, labels)

X_train_prep = preprocess_images(X_train)
X_val_prep = preprocess_images(X_val)
X_test_prep = preprocess_images(X_test)

plot_samples(X_train_prep, y_train, labels)

save_new_images(X_train_prep, y_train, folder_name=TRAIN_DIR)
save_new_images(X_val_prep, y_val, folder_name=VAL_DIR)
save_new_images(X_test_prep, y_test, folder_name=TEST_DIR)

clean_directory(TRAIN_DIR)
clean_directory(VAL_DIR)
clean_directory(TEST_DIR)

model = create_model()

model.summary()

train_generator, validation_generator = create_data_generators()

history = model.fit(
    train_generator,
    steps_per_epoch=6,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=3,
    callbacks=[EarlyStopping(monitor="val_accuracy", mode="max", patience=6)],
)

plot_model_performance(history)

predictions = [1 if x > 0.5 else 0 for x in model.predict(X_test_prep)]

accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy = %.2f" % accuracy)

confusion_mtx = confusion_matrix(y_test, predictions)
plot_confusion_matrix(confusion_mtx, classes=list(labels.items()), normalize=False)