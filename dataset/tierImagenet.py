import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T


class tierImagenet_pretrain(Dataset):
    def __init__(self, resize, labels, images,
                 augment='0', split='train', percentage=0.1):

        self.resize = resize
        self.percentage = percentage  # how many percentage as test from the whole train
        self.split = split

        if augment == '0':
            self.transform = T.Compose([
                # lambda x: Image.open(x).convert('RGB'),
                lambda ind: Image.fromarray(self.data[ind]).convert('RGB'),
                T.Resize((self.resize, self.resize)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif augment == '1':
            if self.split == 'train':
                self.transform = T.Compose([
                    # lambda x: Image.open(x).convert('RGB'),
                    lambda ind: Image.fromarray(self.data[ind]).convert('RGB'),
                    T.Resize((self.resize + 20, self.resize + 20)),
                    T.RandomCrop(self.resize),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                self.transform = T.Compose([
                    # lambda x: Image.open(x).convert('RGB'),
                    lambda ind: Image.fromarray(self.data[ind]).convert('RGB'),
                    T.Resize((self.resize + 20, self.resize + 20)),
                    T.RandomCrop(self.resize),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

        self.cls_num = len(np.unique(labels[0]))

        self.data = []
        self.target = []
        total_sample = 0
        for i in range(self.cls_num):

            curr_sample_pool = images[labels[0] == i]
            total_num = len(curr_sample_pool)
            if self.split == 'train':
                im_data = curr_sample_pool[:int(total_num * self.percentage)]
            else:
                im_data = curr_sample_pool[int(total_num * (1 - self.percentage)):]
            self.data.extend(im_data)

            curr_target_list = [i for _ in range(len(im_data))]
            self.target.extend(curr_target_list)
            total_sample += len(im_data)

        self.total_sample = total_sample

    def __getitem__(self, index):
        """index means index of sets, 0<= index < len(self) """
        x = self.transform(index)
        y = torch.ones(1) * self.target[index]
        y = y.long()

        return x, y

    def __len__(self):
        return len(self.target)


class tierImagenet(Dataset):
    """directly refactored from the miniImagenet.py file"""
    def __init__(self, n_way, k_shot, k_query, resize,
                 labels, images,
                 augment='0', split='train', test=None):

        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.resize = resize
        self.split = split
        if test is not None:
            self.test_manner = test.manner
            if self.test_manner == 'standard':
                self.test_ep_num = test.ep_num
                self.test_query_num = test.query_num

        self._label_specific = labels[0]
        self._label_general = labels[1]
        self._label_specific_str = labels[2]
        self._label_general_str = labels[3]
        self.labels_str = self._label_specific_str
        self.labels = self._label_specific
        self.images = images

        if augment == '0':
            self.transform = T.Compose([
                lambda ind: Image.fromarray(self.images[ind]).convert('RGB'),   # self.images[ind]: array, 0-255
                T.Resize((self.resize, self.resize)),
                T.ToTensor(),
                # TODO (low): comptute im std and mean on tier-imagenet
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif augment == '1':
            if self.split == 'train':
                self.transform = T.Compose([
                    lambda ind: Image.fromarray(self.images[ind]).convert('RGB'),
                    T.Resize((self.resize+20, self.resize+20)),
                    T.RandomCrop(self.resize),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
            else:
                self.transform = T.Compose([
                    lambda ind: Image.fromarray(self.images[ind]).convert('RGB'),
                    T.Resize((self.resize + 20, self.resize + 20)),
                    T.RandomCrop(self.resize),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

        self.cls_num = len(np.unique(self.labels))
        self.support_sz = self.n_way * self.k_shot      # num of samples per support set
        self.query_sz = self.n_way * self.k_query       # num of samples per query set

        self.support_x_batch, self.query_x_batch = [], []
        self.support_y_batch, self.query_y_batch = [], []
        self._create_batch()

    def _create_batch(self):
        """create batch for meta-learning in ONE episode."""
        for _ in range(len(self)):  # for each batch

            # 1. select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            support_x, query_x = [], []
            support_y, query_y = [], []

            # 2. select k_shot + k_query for each class
            if self.split == 'test' and self.test_manner == 'standard':
                assert self.k_query == self.test_query_num

            for cls in selected_cls:
                selected_imgs_idx = np.random.choice(
                    np.nonzero(self.labels == cls)[0], self.k_shot + self.k_query, False)
                indexDtrain = selected_imgs_idx[:self.k_shot]     # idx for Dtrain
                indexDtest = selected_imgs_idx[self.k_shot:]      # idx for Dtest

                support_x.extend(indexDtrain)
                query_x.extend(indexDtest)
                support_y.extend(np.array([cls for _ in range(self.k_shot)]))
                query_y.extend(np.array([cls for _ in range(self.k_query)]))

            self.support_x_batch.append(support_x)
            self.query_x_batch.append(query_x)
            self.support_y_batch.append(support_y)
            self.query_y_batch.append(query_y)

    def __getitem__(self, index):
        """index means index of sets, 0<= index < len(self) """
        support_y = np.array(self.support_y_batch[index])
        query_y = np.array(self.query_y_batch[index])

        support_x = torch.FloatTensor(self.support_sz, 3, self.resize, self.resize)
        query_x = torch.FloatTensor(self.query_sz, 3, self.resize, self.resize)
        for i, curr_ind in enumerate(self.support_x_batch[index]):
            support_x[i] = self.transform(curr_ind)

        for i, curr_ind in enumerate(self.query_x_batch[index]):
            query_x[i] = self.transform(curr_ind)

        return \
            support_x, torch.LongTensor(torch.from_numpy(support_y)), \
            query_x, torch.LongTensor(torch.from_numpy(query_y))

    def __len__(self):
        if self.split == 'val' and self.test_manner == 'standard':
            return self.test_ep_num
        else:
            # return int(np.floor(self.images.shape[0] * 1. / (self.support_sz + self.query_sz)))
            return self.images.shape[0] / (self.support_sz + self.query_sz)

