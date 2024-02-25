import glob
import pickle
import imageio.v3
import os.path
import numpy
import json


class Manager:
    def __init__(self, path):
        self.path = path

    def exist_file(self):
        return os.path.exists(self.path)

    def save_data_pickle(self, data):
        with open(self.path, 'wb') as file:
            pickle.dump(data, file)

    def load_data_pickle(self):
        with open(self.path, 'rb') as file:
            return pickle.load(file)

    def save_data_json(self, data):
        with open(self.path, 'w') as file:
            json.dump(data, file)

    def load_data_json(self):
        with open(self.path, 'r') as file:
            return json.load(file)

    @staticmethod
    def load_mnist_train_data(path):
        with open(path, 'r') as train_data_file:
            return train_data_file.readlines()[1:]

    @staticmethod
    def load_mnist_test_data(path):
        dataset = list()
        with open(path, 'r') as test_data_file:
            for record in test_data_file:
                values = record.split(',')
                answer = int(values[0])
                image_data = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
                dataset.append([answer, image_data])

        return dataset

    @staticmethod
    def load_images_test_data(path):
        dataset = list()
        regular_end_file = '/*.png'
        index_symbol = regular_end_file.index('*') - len(regular_end_file)
        for image_file_name in glob.glob(path + regular_end_file):
            answer = int(image_file_name[index_symbol])
            image = imageio.v3.imread(image_file_name, mode='F')
            image_data = ((255.0 - image.reshape(784)) / 255.0 * 0.99) + 0.01
            dataset.append([answer, image_data])

        return dataset
