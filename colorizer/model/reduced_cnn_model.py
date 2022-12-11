import tensorflow as tf

"""
CNN model architecture produced as a result of reducing the size of the
model proposed by Zhang et al in their paper (reduced for more realistic
compute times given computational resource limitations)
"""
class ReducedCNNModel:
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
        Defines and returns a keras sequential CNN model with a scaled-down
        version of the architecture proposed by Zhang et. al in the original paper.
        """
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(150,150,1)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1, padding="same"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=self.classes_count, kernel_size=1, strides=1, padding="valid", activation="softmax"),
        ])

        return model