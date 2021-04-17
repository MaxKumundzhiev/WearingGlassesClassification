# ------------------------------------------
#
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------

import argparse

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from configurations import IMG_SIZE, BATCH_SIZE, MODEL_PATH


class Inference:
    def prepare_dataset(self, dataset_path):
        test_dataset = tf.keras.preprocessing.image_dataset_from_directory (
            dataset_path,
            shuffle=True,
            seed=123456,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE)
        return test_dataset

    @staticmethod
    def load_model():
        return keras.models.load_model(MODEL_PATH)

    def run(self, path: str):
        positive_classified = []

        test_dataset = self.prepare_dataset(dataset_path=path)
        inference_model = self.load_model()

        image_batch, label_batch = test_dataset.as_numpy_iterator().next()
        predictions = inference_model.predict_on_batch(image_batch).flatten()
        predictions = tf.nn.sigmoid(predictions)
        predictions = tf.where(predictions < 0.5, 0, 1)

        for image_class in predictions:
            if image_class == 1:
                positive_classified.append(image_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, type=str,
                        help='Path to the dataset.')
    arguments = parser.parse_args()

    Inference.run(path=arguments.path)




