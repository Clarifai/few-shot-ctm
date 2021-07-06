import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import numpy as np
from PIL import Image
import csv

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
                if label in list(dictLabels.keys()):
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


