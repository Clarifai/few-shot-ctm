import os
import pickle
import numpy as np

from torch.utils.data import DataLoader
from dataset.tier_imagenet_pretrain import tierImagenet_pretrain
from dataset.mini_imagenet_pretrain import miniImagenet_pretrain
from tools.general_utils import decompress


# for tier-imagenet
def read_data(logger, cache_path):

    # cache_path = os.path.join(self.data_folder, self.split)
    cache_path_labels = cache_path + "_labels.pkl"
    cache_path_images = cache_path + "_images.npz"

    if not os.path.exists(cache_path_images):
        png_pkl = cache_path_images[:-4] + '_png.pkl'
        if os.path.exists(png_pkl):
            decompress(cache_path_images, png_pkl)
        else:
            raise Exception('file not exists! {}'.format(png_pkl))

    assert os.path.exists(cache_path_labels)
    assert os.path.exists(cache_path_images)

    logger("\tRead cached labels from {}".format(cache_path_labels))
    with open(cache_path_labels, "rb") as f:
        data = pickle.load(f)
        label_specific = data[b"label_specific"]
        label_general = data[b"label_general"]
        label_specific_str = data[b"label_specific_str"]
        label_general_str = data[b"label_general_str"]

    logger("\tRead cached images from {}".format(cache_path_images))
    with np.load(cache_path_images, mmap_mode="r", encoding='latin1') as data:
        images = data["images"]
    return [label_specific, label_general, label_specific_str, label_general_str], images


def data_loader_pretrain(opts):

    opts.logger('Preparing datasets for pretraining: [{:s}] ...'.format(opts.dataset.name))
    abs_path = os.getcwd()

    # create data_loader
    if opts.dataset.name == 'mini-imagenet':
        relative_path = 'dataset/miniImagenet/'

        opts.logger('Val data (percentage: {:.2f}) ...'.format(0.1))
        val_data = miniImagenet_pretrain(
            root=os.path.join(abs_path, relative_path), resize=opts.data.im_size,
            augment=opts.data.augment, split='val', percentage=0.1
        )
        opts.logger('\t\tFind {:d} samples'.format(val_data.total_sample))
        opts.logger('\t\tFind {:d} classes'.format(val_data.cls_num))

        opts.logger('Train data (percentage: {:.2f}) ...'.format(0.9))
        train_data = miniImagenet_pretrain(
            root=os.path.join(abs_path, relative_path), resize=opts.data.im_size,
            augment=opts.data.augment, split='train', percentage=0.9
        )
        opts.logger('\t\tFind {:d} samples'.format(train_data.total_sample))
        opts.logger('\t\tFind {:d} classes'.format(train_data.cls_num))

    elif opts.dataset.name == 'tier-imagenet':

        relative_path = 'dataset/tier_imagenet/'
        labels, images = read_data(opts.logger, os.path.join(abs_path, relative_path, 'train'))

        opts.logger('Val data (percentage: {:.2f}) ...'.format(0.1))
        val_data = tierImagenet_pretrain(
            resize=opts.data.im_size, labels=labels, images=images,
            augment=opts.data.augment, split='val', percentage=0.1
        )
        opts.logger('\t\tFind {:d} samples'.format(val_data.total_sample))
        opts.logger('\t\tFind {:d} classes'.format(val_data.cls_num))

        opts.logger('Train data (percentage: {:.2f}) ...'.format(0.9))
        train_data = tierImagenet_pretrain(
            resize=opts.data.im_size, labels=labels, images=images,
            augment=opts.data.augment, split='train', percentage=0.9
        )
        opts.logger('\t\tFind {:d} samples'.format(train_data.total_sample))
        opts.logger('\t\tFind {:d} classes'.format(train_data.cls_num))

    else:
        raise NameError('Unknown dataset ({})!'.format(opts.dataset.name))

    train_db = DataLoader(train_data, opts.train.batch_sz,
                          shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    val_db = DataLoader(val_data, opts.test.batch_sz,
                        shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # update opts
    opts.ctrl.total_iter_train = [len(train_db)] if not opts.ctrl.eager else [100]
    opts.ctrl.total_iter_val = [len(val_db)] if not opts.ctrl.eager else [50]

    return train_db, val_db
