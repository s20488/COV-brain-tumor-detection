from keras import layers
from keras.applications.vgg16 import VGG16
from keras.models import Sequential

from COV_brain_tumor_detection.config import IMG_SIZE


def create_model() -> Sequential:
    base_model = VGG16(
        weights="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
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
