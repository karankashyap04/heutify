import tensorflow as tf
import numpy as np
from scipy import ndimage

class MultinomialCrossEntropyLoss:
    def __init__(self):
        self.LAMBDA = 0.5 # Value suggested in the paper
        self.SIGMA = 5 # Value suggested in the paper
        self.p = None # This will store the ab bin distribution
        self.w = None # Store weighted factors for pixels based on ab bin distribution

        self.Q = 23 ** 2 # We had done 23 x 23 for the Lab colorspace in our model; can change if required
    
    def initialize_pixel_weights(self):
        w = ((1 - self.LAMBDA) * self.p) + (self.LAMBDA / self.Q)
        w = tf.math.reciprocal(w)
        expected_w = np.tensordot(self.p, w)
        output = w / expected_w
        return ndimage.gaussian_filter(input=output, sigma=self.SIGMA)

    def v(self, Z, w):
        # Note: Here, Z is the soft-encoding vector representation of some label
        # Note: Z will be a tensor
        # Note: this computes the v value for a single pixel
        return w[tf.math.argmax(Z)]
    
    def v_image(self, image):
        # Compute v value for an entire image
        v = []
        for row_idx in range(image.shape[0]):
            for col_idx in range(image.shape[1]):
                v.append(self.v(Z=image[row_idx, col_idx, :], w=self.w))
        return tf.convert_to_tensor(v, dtype=tf.float32)

    def loss(self, y_preds, y_true):
        Z = None # TODO: define some soft-encoding scheme to convert each label image to a corresponding ab bin distribution
        v = []
        for image_idx in range(len(Z)):
            v.append(self.v_image(Z[image_idx,:,:,:]))
        v = tf.convert_to_tensor(v, dtype=tf.float32)
        cce = tf.keras.losses.CategoricalCrossentropy()
        crossentropy_loss = cce(Z, y_preds)
        # Not 100% sure of axes on next line... think it should be 3 because the dot product
        # is across q in the formula (it is within the sum across h and w)
        product = tf.tensordot(v, crossentropy_loss, axes=3)
        loss = -product
        return loss