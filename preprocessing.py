import numpy as np
from PIL import Image
from skimage import color
import os

class Preprocessor:
    def __init__(self):
        self.INTEL_BASE_DIR = "data/intel-image-classification-dataset/"
        self.INTEL_SUBDIRECTORIES = ["seg_pred/seg_pred/", "seg_test/seg_test/", "seg_train/seg_train/"]
        self.INTEL_TRAIN_TEST_SUBDIRECTORIES = ["buildings/", "forest/", "glacier/", "mountain/", "sea/", "street/"]
        self.WRITE_DIR = "preprocessed_data/intel-dataset/"

    def get_image_paths(self):
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
        return input_images[:,:,:,0]

    def get_data(self, image_paths):
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
        np.save(self.WRITE_DIR + "inputs", X)
        np.save(self.WRITE_DIR + "labels", y)


if __name__ == '__main__':
    preprocessor = Preprocessor()
    image_paths = preprocessor.get_image_paths()
    X, y = preprocessor.get_data(image_paths=image_paths)
    preprocessor.write_data(X, y)