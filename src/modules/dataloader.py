import json

import numpy as np
from chainer.datasets import TupleDataset, split_dataset_random
from chainer.iterators import SerialIterator
from PIL import Image

class DataLoader():
    def __init__(self, annotation_file_path):
        self.height = 224
        self.width = 224
        self.channel = 3
        self.label_num = 16
        self.annotation_file_path = annotation_file_path
        self.annotation_file = json.load(open((self.annotation_file_path), 'r'))
        self.image_list = self.annotation_file['fileList']
        self.label_list = self.annotation_file['posList']

    def load_data(self):
        x = np.zeros((len(self.image_list), self.height, self.width, self.channel))
        for i, imagepath in enumerate(self.image_list):
            image = Image.open(imagepath).convert('RGB')
            x[i] = np.asarray(image.resize((self.height, self.width))) / 255.
        self.x = x.astype('float32')

        t = np.zeros((len(self.label_list), self.label_num, self.height, self.width))
        for label_idx, label in enumerate(self.label_list):
            for point_idx, point in enumerate(label):
                x_pos = int(point[0] / 600 * 224)
                y_pos = int(point[1] / 600 * 224)
                t[label_idx][point_idx][y_pos][x_pos] = 1
        self.t = t.astype('int32')

        self.dataset = TupleDataset(x, t)
        return self.dataset
        
    def split(self, ratio = (8, 1, 1)):
        train_val_rate = (ratio[0] + ratio[1]) / sum(ratio)
        train_rate = ratio[0] / (ratio[0] + ratio[1])
        train_val, test = split_dataset_random(self.dataset, int(len(self.dataset) * train_val_rate), seed=0)
        train, valid = split_dataset_random(train_val, int(len(train_val) * train_rate), seed=0)
        self.train = train
        self.valid = valid
        self.test = test
        return (self.train, self.valid, self.test)

if __name__ == "__main__":
    dataloader = DataLoader('data/raw/hands/hand_position.json')
    dataloader.load_data()
    train, valid, test = dataloader.split()