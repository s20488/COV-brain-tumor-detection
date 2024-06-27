from keras import layers
from keras.applications.vgg16 import VGG16
from keras.models import Sequential

from COV_brain_tumor_detection.config import IMG_SIZE


def create_model() -> Sequential:
    """
    Creates a VGG16-based CNN for binary classification.

    Returns:
    Sequential: Compiled Keras model instance.

    Modifications:
    - Uses pre-trained VGG16 model without top layers.
    - Adds Flatten, Dropout (0.5), and Dense (sigmoid) layers for classification.
    - Freezes VGG16 base layers to retain pre-trained weights.

    Inspiration:
    https://www.kaggle.com/code/ruslankl/brain-tumor-detection-v1-0-cnn-vgg-16
    """
    base_model = VGG16(
        weights="COV_brain_tumor_detection/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
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
