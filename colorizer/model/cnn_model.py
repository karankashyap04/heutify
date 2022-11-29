import tensorflow as tf

class CNNModel:
    def __init__(self, classes_count: int):
        self.classes_count = classes_count

    def get_cnn_colorizer_model(self):
        # TODO: Might need to change the strides value of all the Conv2DTranspose layers to 1 because the
        # output shapes seem to get quite messed up otherwise
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding="same"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="leaky_relu"),

            # In the Pytorch code associated with the paper, they make two separate models: one for
            # classification and one for regression... we might need to implement the second one too
            # (only done the classification one for now)
            tf.keras.layers.Conv2D(filters=self.classes_count, kernel_size=1, strides=1, padding="valid"),
        ])

        return model