import tensorflow as tf

"""
CNN model architecture produced as a result of reducing the size of the
model proposed by Zhang et al in their paper (reduced for more realistic
compute times given computational resource limitations).
Also, this model's output shape has been modified to make it compatible with
Mean Squared Error (MSE) loss.
"""
class ReducedCNNModelMSE:
    def __init__(self):
        """
        Constructor for the ReducedCNNModelMSE class.
        Does not perform any computations at all.
        """
        pass

    def get_cnn_colorizer_model(self):
        """
        Defines a keras sequential model with a scaled-down version of the
        architecture proposed by Zhang et al in their paper, repurposed to work
        with the MSE loss function.
        """
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(150,150,1)),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1, padding="same"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1, padding="valid"),
        ])

        return model
    
    def mse_loss(truth, pred):
        """
        Computes the mean squared error loss between the true and predicted
        values.

        params:
        truth -> the ground truth values (labels)
        pred -> the values predicted by the model
        """
        loss = tf.reduce_mean(tf.math.square(truth - pred))
        return loss