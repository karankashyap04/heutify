import tensorflow as tf
import numpy as np
from scipy import ndimage
import tensorflow_probability as tfp
tfd = tfp.distributions
import pickle

class MultinomialCrossEntropyLoss:
    def __init__(self, y_true):
        """
        Constructor for the MultinomialCrossEntropyLoss class.
        
        params:
        y_true -> the ground truth labels from the training data
        """
        self.LAMBDA = 0.5 # Value suggested in the paper
        self.SIGMA = 5 # Value suggested in the paper

        self.a_minimum, self.a_maximum = -110, 110
        self.b_minimum, self.b_maximum = -110, 110
        self.a_num_partitions, self.b_num_partitions = 23, 23

        self.Q = self.a_num_partitions * self.b_num_partitions
        self.ab_to_bin_distribution = dict()
        self.initialize_bin_to_ab_array()
        self.initialize_ab_to_bin_distribution()

        # self.p stores the bin distribution for all the bins for every image in
        # the training data
        self.p = tf.reduce_sum(self.get_batch_bin_distribution(y_true), axis=(0,1,2))
        self.w = self.initialize_pixel_weights() # Store weighted factors for pixels based on ab bin distribution
    

    def initialize_ab_to_bin_distribution(self):
        """
        Computes the distribution of bins corresponding to each possible pair
        of discrete a and b values, and assigns this distribution to the (a,b)
        key in the self.ab_to_bin_distribution dictionary.
        Saves this dictionary as 'ab_to_bin_distribution.pkl'.
        """
        for a_idx, a in enumerate(range(self.a_minimum, self.a_maximum + 1)):
            for b_idx, b in enumerate(range(self.b_minimum, self.b_maximum + 1)):
                self.ab_to_bin_distribution[(a,b)] = self.get_pixel_bin_distribution(a,b)
        with open('ab_to_bin_distribution.pkl', 'wb+') as f:
            pickle.dump(self.ab_to_bin_distribution, f)
        # with open('ab_to_bin_distribution.pkl', 'rb') as f:
        #     self.ab_to_bin_distribution = pickle.load(f)

    def initialize_bin_to_ab_array(self):
        """
        Computes the distribution of a and b channel values corresponding to
        each discrete bin that the ab color space has been quantized into.
        Generates a numpy array with these values and saves it as 'bin_to_ab_array.npy'.
        """
        array = []
        for a_bin in range(0, self.a_num_partitions):
            for b_bin in range(0, self.b_num_partitions):
                a, b = self.bin_to_discrete_ab(a_bin, b_bin)
                array.append([a, b])
        array = np.array(array, dtype=np.float32)
        np.save("bin_to_ab_array", array)
    
    def initialize_pixel_weights(self):
        """
        Computes the weights that each pixel will be weighed by. These values
        are stored in self.w.
        """
        w = ((1 - self.LAMBDA) * self.p) + (self.LAMBDA / self.Q)
        w = tf.math.reciprocal(w)
        expected_w = np.dot(self.p, w)
        output = w / expected_w
        return ndimage.gaussian_filter(input=output, sigma=self.SIGMA)

    def ab_to_discrete_bin(self, a, b):
        """
        Takes a single pair of a and b color channel values in the continuous
        color space. Computes and returns the indices of the a_bin and b_bin,
        so that these can be used to index and identify the exact bin the 
        quantized analog of the color space that corresponds to the provided
        a and b color channel values.

        params: 
        a -> 'a' color channel value (in the continuous color space)
        b -> 'b' color channel value (in the continuous color space)
        """
        a_bin = (a - self.a_minimum) / (self.a_maximum - self.a_minimum) * (self.a_num_partitions - 1)
        b_bin = (b - self.b_minimum) / (self.b_maximum - self.b_minimum) * (self.b_num_partitions - 1)
        return round(a_bin), round(b_bin)
    
    def bin_to_discrete_ab(self, a_bin, b_bin):
        """
        Takes a single pair of a_bin and b_bin indices, which can be used to index
        a single discrete bin in the quantized color space. Computes and returns
        the a and b color channel values in the continuous color space that
        correspond to the provided bin ids.

        params:
        a_bin -> the index of the a bin
        b_bin -> the index of the b bin
        """
        # For a single pixel
        a = a_bin * (self.a_maximum - self.a_minimum) / self.a_num_partitions + self.a_minimum
        b = b_bin * (self.b_maximum - self.b_minimum) / self.b_num_partitions + self.b_minimum
        return a, b
    
    def get_nearest_discrete_ab(self, a, b):
        """
        Takes a single pair of a and b color channel values in the continuous
        color space. Finds the bins in the discretized analog of the color space
        that are the 5 nearest neighbors to the provided point coordinates in
        the continuous ab color space.
        Returns a list (sorted in increasing order) containing the distances
        of these bins from the point in the continuous space (when the point is
        mapped onto the discrete space), the a and b values corresponding to each
        of these bins, and the a and b bin indices of each of these bins.

        params:
        a -> 'a' color channel values (in the continuous color space)
        b -> 'b' color channel values (in the continuous color space)
        """
        nearest_abs = []
        main_bin = self.ab_to_discrete_bin(a, b)
        central_a_bin, central_b_bin = main_bin

        if central_a_bin == 0:
            a_lower, a_upper = 0, 2
        elif central_a_bin == self.a_num_partitions - 1:
            a_lower, a_upper = central_a_bin - 2, central_a_bin
        else:
            a_lower, a_upper = central_a_bin - 1, central_a_bin + 1
        
        if central_b_bin == 0:
            b_lower, b_upper = 0, 2
        elif central_b_bin == self.b_num_partitions - 1:
            b_lower, b_upper = central_b_bin - 2, central_b_bin
        else:
            b_lower, b_upper = central_b_bin - 1, central_b_bin + 1
        
        for a_idx in range(a_lower, a_upper + 1):
            for b_idx in range(b_lower, b_upper + 1):
                a_val, b_val = self.bin_to_discrete_ab(a_idx, b_idx)
                dist = np.linalg.norm(np.array((a_val, b_val)) - np.array((a, b)))
                nearest_abs.append((dist, (a_val, b_val), (a_idx, b_idx)))
        
        nearest_abs.sort(key=lambda x: x[0]) # Sort based on distances
        return nearest_abs[:5] # Returning 5 nearest points along with distances and bin ids
    
    def get_pixel_bin_distribution(self, a, b): # soft encoding scheme for one pixel
        """
        Defines the soft encodign scheme for a single pixel.
        Takes the a and b color channel values (in the continuous color space)
        for a single pixel, and computes a vector representation for this pixel
        with in the same number of dimensions as the number of discrete bins
        (i.e. converts the pixel's a and b values to a bin distribution using
        the 5 nearest bin neighbors to the provided a and b coordinates, by
        weighing values according to distances using a Gaussian kernel).

        params:
        a -> 'a' color channel values (in the continuous color space)
        b -> 'b' color channel values (in the continuous color space)
        """
        a, b = round(a), round(b)
        nearest_abs = self.get_nearest_discrete_ab(a, b)
        gaussian_distribution = tfd.Normal(loc=0, scale=5.)
        pixel_bin_distribution = tf.zeros(shape=(self.a_num_partitions, self.b_num_partitions), dtype=tf.float32)
        pixel_bin_distribution_np = pixel_bin_distribution.numpy()
        for dist, _, bin_ids in nearest_abs:
            pixel_bin_distribution_np[bin_ids[0]][bin_ids[1]] = gaussian_distribution.prob(dist)
        pixel_bin_distribution = tf.convert_to_tensor(pixel_bin_distribution_np)
        pixel_bin_distribution = tf.reshape(pixel_bin_distribution, shape=(self.Q))
        self.ab_to_bin_distribution[(a,b)] = pixel_bin_distribution
        return pixel_bin_distribution
    
    def get_image_bin_distribution(self, image):
        """
        Takes an image and computes the bin distribution for every pixel of the
        image (a distribution of values is computed across all the discrete bins
        for each pixel). Returns this bin distribution.

        params:
        image -> a ground truth image (i.e. an image from y_true) - this image
                contains values for the a and b channels for each of its pixels.
        """
        flattened_image = tf.reshape(image, shape=(-1, 2))
        image_bin_distribution = tf.map_fn(lambda pixel: self.get_pixel_bin_distribution(pixel[0].numpy(), pixel[1].numpy()), elems=flattened_image)
        image_bin_distribution = tf.reshape(image_bin_distribution, shape=(image.shape[0], image.shape[1], self.Q))
        return image_bin_distribution
    
    def get_batch_bin_distribution(self, y_true):
        """
        Computes the bin distribution for every image in y_true (i.e. for every
        pixel of every image, the a and b color channel values are taken and
        converted into a distribution over all of the discrete bins in the
        quantized color space). This batch bin distribution is returned.
        """
        batch_bin_distribution = []
        for image in y_true:
            image_bin_distribution = self.get_image_bin_distribution(image=image)
            batch_bin_distribution.append(image_bin_distribution)
        batch_bin_distribution = tf.convert_to_tensor(batch_bin_distribution, dtype=tf.float32)
        return batch_bin_distribution

    def v_pixel(self, Z, w):
        """
        Computes the value of the v function (from the original paper) on a
        single pixel. This is effectively used as a reweighting term that
        aims to prevent the predicted a and b values in the image from all being
        clustered around a single range.

        params:
        Z -> a soft-encoded vector representation of a single pixel from some
            ground truth image
        w -> the distribution of ab values across all the discrete bins which
            was initially computed using all of the training labels (stored in
            self.w)
        """
        return w[tf.math.argmax(Z)]
    
    def v_image(self, image):
        """
        Computes the value of the v function (from the original paper) on a
        single image. This is used as a reweighting term that aims to prevent
        the predicted a and b values in the image from all being clustered around
        a single range. This is done by slightly increasing the loss of pixels
        with a and b values that are common, and decreasing the loss of pixels
        with a and b values that are more rare.

        params:
        image -> the soft-encoded vector representation of some ground truth image.
        """
        v = []
        for row_idx in range(image.shape[0]):
            for col_idx in range(image.shape[1]):
                v.append(self.v_pixel(Z=image[row_idx, col_idx, :], w=self.w))
        result = tf.convert_to_tensor(v, dtype=tf.float32)
        result = tf.reshape(result, shape=(image.shape[0], image.shape[1]))
        return result

    def loss(self, y_true, y_preds):
        """
        Computes the multinomial cross-entropy loss.

        params:
        y_true -> the ground truth labels for the a and b channel values for an image
        y_preds -> the model output (a probability distribution across all of the
                    discrete bins for each pixel in the image).
        """
        Z = self.get_batch_bin_distribution(y_true)
        v = []
        for image_idx in range(len(Z)):
            v.append(self.v_image(Z[image_idx,:,:,:]))
        v = tf.convert_to_tensor(v, dtype=tf.float32)
        crossentropy_loss = tf.keras.losses.categorical_crossentropy(Z, y_preds)
        product = tf.tensordot(v, crossentropy_loss, axes=3)
        loss = -product
        return loss