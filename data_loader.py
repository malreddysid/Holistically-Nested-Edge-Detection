import numpy as np
import PIL.Image as Image

class DataLoader():
    def __init__(self, data_path, file_list):
        self.data_path = data_path
        self.file_list = file_list
        with open(self.file_list, 'r') as f:
            self.train = f.readlines()
        self.train = [x.split() for x in self.train]
        self.num_train = len(self.train)
        self.order = np.random.permutation(self.num_train)

    def shuffle_data(self):
        self.order = np.random.permutation(self.num_train)

    def get_data(self, i):
        idx = self.order[i]

        img = Image.open(self.data_path + self.train[idx][0])
        img = np.asarray(img)
        img.setflags(write=1)
        img[:, :, 0] = img[:, :, 0] - 122.67892
        img[:, :, 1] = img[:, :, 1] - 116.66877
        img[:, :, 2] = img[:, :, 2] - 104.00699
        img = np.tile(img, [1, 1, 1, 1])

        lb = Image.open(self.data_path + self.train[idx][1]).convert('L')
        lb = np.asarray(lb)
        lb.setflags(write=1)
        lb[lb > 0] = 1
        lb = np.expand_dims(lb, axis=0)
        lb = np.expand_dims(lb, axis=3)

        return img, lb

    def get_num_train(self):
        return self.num_train
