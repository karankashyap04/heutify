import tensorflow as tf
import numpy as np
from scipy import ndimage
import tensorflow_probability as tfp
tfd = tfp.distributions
import pickle

class MultinomialCrossEntropyLoss:
    def __init__(self, y_true):
        self.LAMBDA = 0.5 # Value suggested in the paper
        self.SIGMA = 5 # Value suggested in the paper

        # TODO: Fill these out
        # self.a_minimum, self.a_maximum = None, None
        # self.b_minimum, self.b_maximum = None, None
        self.a_minimum, self.a_maximum = -110, 110
        print("self.a_minimum:", self.a_minimum)
        self.b_minimum, self.b_maximum = -110, 110
        # self.a_num_partitions, self.b_num_partitions = None, None # I think each of these should be 23 so that self.Q can be self.a_num_partitions * self.b_num_partitions
        self.a_num_partitions, self.b_num_partitions = 23, 23

        self.Q = 23 ** 2 # We had done 23 x 23 for the Lab colorspace in our model; can change if required
        self.ab_to_bin_distribution = dict()
        self.initialize_ab_to_bin_distribution()

        # TODO: Fill these out
        # self.p = None # This will store the bin distribution for all the bins
        # for every image in the training data (can probably do this using the get_batch_bin_distribution function)
        self.p = tf.reduce_sum(self.get_batch_bin_distribution(y_true), axis=(0,1,2))
        print("begin self.w initialization")
        self.w = self.initialize_pixel_weights() # Store weighted factors for pixels based on ab bin distribution
    

    def initialize_ab_to_bin_distribution(self):
        print("Start")
        # for a_idx, a in enumerate(range(self.a_minimum, self.a_maximum + 1)):
        #     for b_idx, b in enumerate(range(self.b_minimum, self.b_maximum + 1)):
        #         self.ab_to_bin_distribution[(a,b)] = self.get_pixel_bin_distribution(a,b)
        #     print(f"Completed {a_idx}")
        # print("Finish")
        # print("Write to pkl file:")
        # with open('ab_to_bin_distribution.pkl', 'wb+') as f:
        #     pickle.dump(self.ab_to_bin_distribution, f)
        # print("Done writing file")
        with open('ab_to_bin_distribution.pkl', 'rb') as f:
            self.ab_to_bin_distribution = pickle.load(f)
        print("Done loading file")
    
    def initialize_pixel_weights(self):
        w = ((1 - self.LAMBDA) * self.p) + (self.LAMBDA / self.Q)
        w = tf.math.reciprocal(w)
        print("w shape:", w.shape)
        print("self.p shape:", self.p.shape)
        # expected_w = np.tensordot(self.p, w)
        expected_w = np.dot(self.p, w)
        output = w / expected_w
        return ndimage.gaussian_filter(input=output, sigma=self.SIGMA)


    def ab_to_discrete_bin(self, a, b):
        # Convert single pixel a, b to bin id (a_bin, b_bin)
        a_bin = (a - self.a_minimum) / (self.a_maximum - self.a_minimum) * (self.a_num_partitions - 1)
        b_bin = (b - self.b_minimum) / (self.b_maximum - self.b_minimum) * (self.b_num_partitions - 1)
        return round(a_bin), round(b_bin)
        # return round(a_bin.numpy()), round(b_bin.numpy())
    
    def bin_to_discrete_ab(self, a_bin, b_bin):
        # For a single pixel
        a = a_bin * (self.a_maximum - self.a_minimum) / self.a_num_partitions + self.a_minimum
        b = b_bin * (self.b_maximum - self.b_minimum) / self.b_num_partitions + self.b_minimum
        return a, b
    
    def get_nearest_discrete_ab(self, a, b):
        nearest_abs = []
        main_bin = self.ab_to_discrete_bin(a, b)
        central_a_bin, central_b_bin = main_bin
        # nearest_abs.append(self.bin_to_discrete_ab(central_a_bin, central_b_bin))

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
                # if a_idx == central_a_bin and b_idx == central_b_bin:
                #     continue
                a_val, b_val = self.bin_to_discrete_ab(a_idx, b_idx)
                dist = np.linalg.norm(np.array((a_val, b_val)) - np.array((a, b)))
                nearest_abs.append((dist, (a_val, b_val), (a_idx, b_idx)))
        
        nearest_abs.sort(key=lambda x: x[0]) # Sort based on distances
        return nearest_abs[:5] # Returning 5 nearest points along with distances and bin ids
    
    def get_pixel_bin_distribution(self, a, b):
        a, b = round(a), round(b)
        nearest_abs = self.get_nearest_discrete_ab(a, b)
        gaussian_distribution = tfd.Normal(loc=0, scale=5.)
        pixel_bin_distribution = tf.zeros(shape=(self.a_num_partitions, self.b_num_partitions), dtype=tf.float32)
        # pixel_bin_distribution = tf.Variable(pixel_bin_distribution, trainable=False)
        pixel_bin_distribution_np = pixel_bin_distribution.numpy()
        for dist, _, bin_ids in nearest_abs:
            pixel_bin_distribution_np[bin_ids[0]][bin_ids[1]] = gaussian_distribution.prob(dist)
            # pixel_bin_distribution = pixel_bin_distribution[bin_ids[0]][bin_ids[1]].assign(gaussian_distribution.prob(dist))
        pixel_bin_distribution = tf.convert_to_tensor(pixel_bin_distribution_np)
        pixel_bin_distribution = tf.reshape(pixel_bin_distribution, shape=(self.Q))
        self.ab_to_bin_distribution[(a,b)] = pixel_bin_distribution
        return pixel_bin_distribution
    
    # @tf.function
    def get_image_bin_distribution(self, image):
        print("begin image bin distribution")
        print("image shape:", image.shape)
        flattened_image = tf.reshape(image, shape=(-1, 2))
        print("flattened image shape:", flattened_image.shape)
        print("pixel[0]", flattened_image[0][0])
        image_bin_distribution = tf.map_fn(lambda pixel: self.get_pixel_bin_distribution(pixel[0].numpy(), pixel[1].numpy()), elems=flattened_image)
        print("image_bin_distribution shape:", image_bin_distribution.shape)
        # print("desired shape:", (-1, image.shape[0], image.shape[1], image))
        # image_bin_distribution = tf.reshape(image_bin_distribution, shape=image.shape)
        # image_bin_distribution = tf.reshape(image_bin_distribution, shape=(-1, image.shape[0], image.shape[1], self.Q))
        image_bin_distribution = tf.reshape(image_bin_distribution, shape=(image.shape[0], image.shape[1], self.Q))
        # image_bin_distribution = tf.reduce_sum(image_bin_distribution, axis=(0,1,2))
        return image_bin_distribution
    
    def get_batch_bin_distribution(self, y_true):
        print("begin get batch bin distribution")
        batch_bin_distribution = []
        for image in y_true:
            print("image shape: ", image.shape)
            image_bin_distribution = self.get_image_bin_distribution(image=image)
            batch_bin_distribution.append(image_bin_distribution)
        batch_bin_distribution = tf.convert_to_tensor(batch_bin_distribution, dtype=tf.float32)
        print("finish batch bin distribution")
        print("batch bin distribution shape:", batch_bin_distribution.shape)
        return batch_bin_distribution

    def v_pixel(self, Z, w):
        # Note: Here, Z is the soft-encoding vector representation of some label
        # Note: Z will be a tensor
        # Note: this computes the v value for a single pixel
        return w[tf.math.argmax(Z)]
    
    def v_image(self, image):
        # Compute v value for an entire image
        print("v_image image shape:", image.shape)
        v = []
        for row_idx in range(image.shape[0]):
            for col_idx in range(image.shape[1]):
                v.append(self.v_pixel(Z=image[row_idx, col_idx, :], w=self.w))
        result = tf.convert_to_tensor(v, dtype=tf.float32)
        print("result shape:", result.shape)
        result = tf.reshape(result, shape=(image.shape[0], image.shape[1]))
        print("reshaped result shape:", result.shape)
        return result

    def loss(self, y_true, y_preds):
        # Might want to do y_true = y_true[:,:,:,1:] to remove L and have only a, b
        # ADDING A NEW LINE HERE FOR NOW:
        # self.p = self.get_batch_bin_distribution(y_true)
        # y_true_ab = y_true[:,:,:,1:]
        print("y_preds shape:", y_preds.shape)
        Z = self.get_batch_bin_distribution(y_true)
        v = []
        print("Z shape:", Z.shape)
        for image_idx in range(len(Z)):
            v.append(self.v_image(Z[image_idx,:,:,:]))
        v = tf.convert_to_tensor(v, dtype=tf.float32)
        # cce = tf.keras.losses.CategoricalCrossentropy()
        # crossentropy_loss = cce(Z, y_preds)
        crossentropy_loss = tf.keras.losses.categorical_crossentropy(Z, y_preds)
        # Not 100% sure of axes on next line... think it should be 3 because the dot product
        # is across q in the formula (it is within the sum across h and w)
        print("v shape:", v.shape)
        print("crossentropy_loss shape:", crossentropy_loss.shape)
        product = tf.tensordot(v, crossentropy_loss, axes=3)
        loss = -product
        return loss