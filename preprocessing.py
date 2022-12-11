import numpy as np
from PIL import Image
from skimage import color
import os

"""
Class containing the code used to preprocess the images from the Intel Image
Classification Dataset.
"""
class Preprocessor:
    def __init__(self):
        """
        Constructor for the Preprocessor class.
        Used to initialize filepaths needed for the preprocessor to work.
        """
        self.INTEL_BASE_DIR = "data/intel-image-classification-dataset/"
        self.INTEL_SUBDIRECTORIES = ["seg_pred/seg_pred/", "seg_test/seg_test/", "seg_train/seg_train/"]
        self.INTEL_TRAIN_TEST_SUBDIRECTORIES = ["buildings/", "forest/", "glacier/", "mountain/", "sea/", "street/"]
        self.WRITE_DIR = "preprocessed_data/intel-dataset/"

    def get_image_paths(self):
        """
        Creates a list of filepaths for all images to be preprocessed based on 
        the folder filepaths initialized in the constructor.
        """
        image_paths = []
        for subdirectory in self.INTEL_SUBDIRECTORIES:
            if subdirectory == "seg_pred/seg_pred/":
                SUBDIRECTORY_PATH = self.INTEL_BASE_DIR + subdirectory
                image_paths += map(lambda path: SUBDIRECTORY_PATH + path, os.listdir(SUBDIRECTORY_PATH))
            else:
                for class_subdirectory in self.INTEL_TRAIN_TEST_SUBDIRECTORIES:
                    SUBDIRECTORY_PATH = self.INTEL_BASE_DIR + subdirectory + class_subdirectory
                    image_paths += map(lambda path: SUBDIRECTORY_PATH + path, os.listdir(SUBDIRECTORY_PATH))
        return image_paths

    def preprocess_images(self, input_images):
        """
        Preprocesses a numpy array of images represented in the Lab colorspace by
        isolating the L channel and returning it (returns the images with the
        a and b channel values removed).

        params:
        input_images -> a numpy array of image data represented in the Lab
                        color space.
        """
        return input_images[:,:,:,0]

    def get_data(self, image_paths):
        """
        Reads the data from all the provided image filepaths. This image data
        is converted from the RGB color space to the Lab colorspace. Then the
        data is preprocessed and the preprocessed inputs and labels are returned.

        params:
        image_paths -> a list of filepaths to all the images whose data needs
                        to be read and preprocessed.
        """
        X = []
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path)
            np_image = np.asarray(image) / float(255) # Normalize the image
            if np_image.shape != (150, 150, 3):
                continue
            lab_image = color.rgb2lab(np_image)
            X.append(lab_image)
            if i % 100 == 0:
                print(f"Completed {i} out of {len(image_paths)}")
        X = np.asarray(X, dtype=np.float32)
        y = np.copy(X)
        return self.preprocess_images(input_images=X), y
    
    def write_data(self, X, y):
        """
        Takes the preprocessed input and label data (two numpy arrays) and saves
        this data as .npy files (within the directory specified in self.WRITE_DIR).

        params:
        X -> a numpy array containing the input data (black and white images i.e.
                images with only the L channel)
        y -> a numpy array containing the ground truth images
        """
        np.save(self.WRITE_DIR + "inputs", X)
        np.save(self.WRITE_DIR + "labels", y)


if __name__ == '__main__':
    """
    Main method for this file. Used to instantiate the Preprocessor class and
    run the code in it to preprocess the images.
    """
    preprocessor = Preprocessor()
    image_paths = preprocessor.get_image_paths()
    X, y = preprocessor.get_data(image_paths=image_paths)
    preprocessor.write_data(X, y)