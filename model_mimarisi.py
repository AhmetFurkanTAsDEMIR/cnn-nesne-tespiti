# Dosya Adı: model_mimarisi.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from veri_hazirlama import IMG_SIZE, NUM_CLASSES

LEARNING_RATE = 1e-3


def build_cnn_detector():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # 1. Evrişim Bloğu
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 2. Evrişim Bloğu
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 3. Evrişim Bloğu
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 4. Evrişim Bloğu
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 5. Evrişim Bloğu
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Çıkış 1: Nesne Sınıfı Tahmini (Araba, Kedi, Köpek, Motor, Otobüs, İnsan)
    cls_branch = layers.Dense(256, activation="relu")(x)
    cls_branch = layers.Dropout(0.4)(cls_branch)
    class_output = layers.Dense(NUM_CLASSES, activation="softmax", name="class_output")(cls_branch)

    # Çıkış 2: Bounding Box Koordinat Tahmini ([xmin, ymin, xmax, ymax])
    bbox_branch = layers.Dense(128, activation="relu")(x)
    bbox_branch = layers.Dense(64, activation="relu")(bbox_branch)
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox_output")(bbox_branch)

    model = models.Model(inputs=inputs, outputs=[class_output, bbox_output], name="CNN_Object_Detector")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=["categorical_crossentropy", "mse"],
        loss_weights=[1.0, 10.0],
        metrics=[["accuracy"], ["mae"]]
    )
    return model


if __name__ == "__main__":
    m = build_cnn_detector()
    m.summary()