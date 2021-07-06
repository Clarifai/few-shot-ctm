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


