import tensorflow as tf

"""
CNN model with the complete architecture proposed in the original paper by Zhang et. al
"""
class CNNModel:
    def __init__(self, classes_count: int=529):
        """
        Constructor for the CNNModel class.

        params:
        classes_count -> the number of discrete bins that the continuous ab
                        color space is being divided into.
        """
        self.classes_count = classes_count

    def get_cnn_colorizer_model(self):
        """
        Defines and returns a keras sequential CNN model with the architecture
        proposed by Zhang et. al in the original paper.
        """
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(150,150,1)),
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

            tf.keras.layers.Conv2D(filters=self.classes_count, kernel_size=1, strides=1, padding="valid"),
        ])

        return model