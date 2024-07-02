from keras.applications import VGG16, EfficientNetB0, NASNetMobile
from tensorflow.keras import Sequential, layers

from COV_brain_tumor_detection.config import IMG_SIZE


def create_model_vgg16() -> Sequential:
    """
    Creates a VGG16-based CNN for binary classification.

    Returns:
    Sequential: Compiled Keras models instance.

    Modifications:
    - Uses pre-trained VGG16 models without top layers.
    - Adds Flatten, Dropout (0.5), and Dense (sigmoid) layers for classification.
    - Freezes VGG16 base layers to retain pre-trained weights.

    Inspiration:
    https://www.kaggle.com/code/ruslankl/brain-tumor-detection-v1-0-cnn-vgg-16
    """
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,),
    )
    model = Sequential(
        [
            base_model,
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.layers[0].trainable = False

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model


def create_model_efficientnet() -> Sequential:
    """
    Creates an EfficientNetB0-based CNN for binary classification.

    Returns:
    Sequential: Compiled Keras models instance.

    Modifications:
    - Uses pre-trained EfficientNetB0 model without top layers.
    - Adds GlobalAveragePooling2D, Dropout (0.5), and Dense (sigmoid) layers for classification.
    - Freezes EfficientNetB0 base layers to retain pre-trained weights.
    """
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
    )
    model = Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    base_model.trainable = False

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model


def create_model_nasnetmobile() -> Sequential:
    """
    Creates a NASNetMobile-based CNN for binary classification.

    Returns:
    Sequential: Compiled Keras models instance.

    Modifications:
    - Uses pre-trained NASNetMobile model without top layers.
    - Adds GlobalAveragePooling2D, Dropout (0.5), and Dense (sigmoid) layers for classification.
    - Freezes NASNetMobile base layers to retain pre-trained weights.
    """
    base_model = NASNetMobile(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
    )
    model = Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    base_model.trainable = False

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model
