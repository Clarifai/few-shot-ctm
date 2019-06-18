import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import numpy as np
from PIL import Image
import csv


class miniImagenet(Dataset):
    """
    put mini-imagenet files as:
        root :
        |- images/*.jpg includes all images
        |- train.csv
        |- test.csv
        |- val.csv

    NOTICE:
    meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: contains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
"""

    def __init__(self, root, n_way, k_shot, k_query,
                 resize, split, augment='0', test=None, shuffle=True):

        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.resize = resize
        self.split = split
        self.shuffle = shuffle
        if test is not None:
            self.test_manner = test.manner
            self.test_ep_num = test.ep_num
            self.test_query_num = test.query_num

        if augment == '0':
            self.transform = T.Compose([
                lambda x: Image.open(x).convert('RGB'),
                T.Resize((self.resize, self.resize)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif augment == '1':
            if self.split == 'train':
                self.transform = T.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    T.Resize((self.resize+20, self.resize+20)),
                    T.RandomCrop(self.resize),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                self.transform = T.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    T.Resize((self.resize + 20, self.resize + 20)),
                    T.RandomCrop(self.resize),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

        self.path = os.path.join(root, 'images')
        csvdata = self._loadCSV(os.path.join(root, self.split + '.csv'))
        self.data = []          # store images for each class [[img1, img2, ...], [img111, img222, ...]]
        self.img2label = {}     # {"img_name[:9]": label}

        total_sample = 0
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)
            self.img2label[k] = i
            total_sample += len(v)

        self.total_sample = total_sample
        self.cls_num = len(self.data)
        self.support_sz = self.n_way * self.k_shot  # num of samples per support set
        self.query_sz = self.n_way * self.k_query

        self.support_x_batch = []
        self.query_x_batch = []
        self._create_batch()

    @staticmethod
    def _loadCSV(csvf):
        """
        return a dict saving the information of csv
        :param csvf: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def _create_batch(self):

        for _ in range(len(self)):  # for each batch

            # 1. select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            support_x, query_x = [], []

            # 2. select k_shot + k_query for each class
            if self.split == 'test' and self.test_manner == 'standard':
                assert self.k_query == self.test_query_num

            for cls in selected_cls:

                selected_imgs_idx = np.random.choice(
                    len(self.data[cls]), self.k_shot + self.k_query, False)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])

                # get all images filename for current Dtrain and Dtest
                support_x.append(np.array(self.data[cls])[indexDtrain].tolist())
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """index means index of sets, 0<= index < len(self)"""
        flatten_support_x_path = [
            os.path.join(self.path, item)
            for cls in self.support_x_batch[index] for item in cls
        ]
        support_y = np.array([
            # item: n0153282900000005.jpg, the first 9 characters treated as label
            self.img2label[item[:9]]
            for cls in self.support_x_batch[index] for item in cls
        ])
        flatten_query_x_path = [
            os.path.join(self.path, item)
            for sublist in self.query_x_batch[index] for item in sublist
        ]
        query_y = np.array([
            self.img2label[item[:9]]
            for sublist in self.query_x_batch[index] for item in sublist
        ])

        support_x = torch.FloatTensor(self.support_sz, 3, self.resize, self.resize)
        query_x = torch.FloatTensor(self.query_sz, 3, self.resize, self.resize)
        for i, path in enumerate(flatten_support_x_path):
            support_x[i] = self.transform(path)
        for i, path in enumerate(flatten_query_x_path):
            query_x[i] = self.transform(path)

        if self.shuffle:
            index = np.random.permutation(len(query_y))
            query_y = query_y[index]
            query_x = query_x[index]

        return \
            support_x, torch.LongTensor(torch.from_numpy(support_y)), \
            query_x, torch.LongTensor(torch.from_numpy(query_y))

    def __len__(self):
        if self.split == 'test' and self.test_manner == 'standard':
            return self.test_ep_num
        else:
            return int(self.total_sample / (self.support_sz + self.query_sz))


class miniImagenet_pretrain(Dataset):
    """
    put mini-imagenet files as:
        root :
        |- images/*.jpg includes all images
        |- train.csv
        |- test.csv
        |- val.csv

    NOTICE:
    meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: contains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
"""

    def __init__(self, root, resize, percentage=0.1, augment='0', split='train'):

        self.resize = resize
        self.percentage = percentage   # how many percentage as test from the whole train
        self.split = split

        if augment == '0':
            self.transform = T.Compose([
                lambda x: Image.open(x).convert('RGB'),
                T.Resize((self.resize, self.resize)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif augment == '1':
            if self.split == 'train':
                self.transform = T.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    T.Resize((self.resize+20, self.resize+20)),
                    T.RandomCrop(self.resize),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                self.transform = T.Compose([
                    lambda x: Image.open(x).convert('RGB'),
                    T.Resize((self.resize + 20, self.resize + 20)),
                    T.RandomCrop(self.resize),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

        self.path = os.path.join(root, 'images')
        csvdata = self._loadCSV(os.path.join(root, 'train.csv'))
        self.data = []
        self.target = []
        total_sample = 0
        for i, (k, v) in enumerate(csvdata.items()):

            total_num = len(v)
            if self.split == 'train':
                im_list = v[:int(total_num*self.percentage)]
            else:
                im_list = v[int(total_num*(1-self.percentage)):]
            curr_im_path = [os.path.join(self.path, x) for x in im_list]
            self.data.extend(curr_im_path)
            curr_target_list = [i for _ in im_list]
            self.target.extend(curr_target_list)
            total_sample += len(im_list)

        self.total_sample = total_sample
        self.cls_num = i + 1

    @staticmethod
    def _loadCSV(csvf):
        """
        return a dict saving the information of csv
        :param csvf: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def __getitem__(self, index):
        """index means index of sets, 0<= index < len(self)"""

        x = self.transform(self.data[index])
        y = torch.ones(1)*self.target[index]
        y = y.long()

        return x, y

    def __len__(self):
        return len(self.target)

