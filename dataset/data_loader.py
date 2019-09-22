import os
import pickle
import numpy as np

from torch.utils.data import DataLoader
from dataset.tierImagenet import tierImagenet
from dataset.mini_imagenet import miniImagenet
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


def data_loader(opts):

    train_db, val_db, test_db, trainval_db = [], [], [], []
    opts.logger('Preparing datasets: [{:s}] ...'.format(opts.dataset.name))
    opts.logger('\t\tn_way is {}, k_shot is {}, k_query is {}'.format(
        opts.fsl.n_way, opts.fsl.k_shot, opts.fsl.k_query))

    # create data_loader
    if opts.dataset.name == 'mini-imagenet':

        relative_path = 'dataset/miniImagenet/'

        opts.logger('Test data ...')
        k_query = opts.fsl.k_query[0] if opts.test.manner == 'same_as_train' else opts.test.query_num
        val_data = miniImagenet(
            root=relative_path,
            n_way=opts.fsl.n_way[0], k_shot=opts.fsl.k_shot[0], k_query=k_query,
            resize=opts.data.im_size, augment=opts.data.augment,
            split='test', test=opts.test)
        opts.logger('\t\tFind {:d} samples'.format(val_data.total_sample))
        opts.logger('\t\tFind {:d} classes'.format(val_data.cls_num))
        val_data = [val_data]

        opts.logger('Train data ...')
        train_data = miniImagenet(
            root=relative_path,
            n_way=opts.fsl.n_way[0], k_shot=opts.fsl.k_shot[0], k_query=opts.fsl.k_query[0],
            resize=opts.data.im_size, augment=opts.data.augment,
            split='train', test=opts.test)
        opts.logger('\t\tFind {:d} samples'.format(train_data.total_sample))
        opts.logger('\t\tFind {:d} classes'.format(train_data.cls_num))
        train_data = [train_data]

    elif opts.dataset.name == 'tier-imagenet':

        relative_path = 'dataset/tier_imagenet/'
        # val data
        phase = 'val'
        labels, images = read_data(opts.logger, os.path.join(relative_path, phase))
        val_data = []
        for i in range(len(opts.fsl.n_way)):
            curr_shot = opts.fsl.k_shot[0] if len(opts.fsl.k_shot) == 1 else opts.fsl.k_shot[i]
            if opts.test.manner == 'standard':
                curr_query = opts.test.query_num
            elif opts.test.manner == 'same_as_train':
                curr_query = opts.fsl.k_query[0] if len(opts.fsl.k_query) == 1 else opts.fsl.k_query[i]

            opts.logger('\t\tcreating batch_ids in [{}] mode, '
                        'im_num:{:d}, {:d}-way, {:d}-shot, {:d}-query, im_resize:{:d}'.format(
                            phase.upper(), images.shape[0],
                            opts.fsl.n_way[i], curr_shot, curr_query, opts.data.im_size))
            val_data.append(tierImagenet(
                n_way=opts.fsl.n_way[i], k_shot=curr_shot, k_query=curr_query,
                resize=opts.data.im_size, augment=opts.data.augment,
                images=images, labels=labels, split=phase, test=opts.test
            ))

        # train data
        phase = 'train'
        if not opts.ctrl.eager:
            # update input data
            labels, images = read_data(opts.logger, os.path.join(relative_path, phase))
        else:
            opts.logger('\tNOTE: eager mode, use val_data as train_data ...')

        train_data = []
        for i in range(len(opts.fsl.n_way)):
            curr_shot = opts.fsl.k_shot[0] if len(opts.fsl.k_shot) == 1 else opts.fsl.k_shot[i]
            curr_query = opts.fsl.k_query[0] if len(opts.fsl.k_query) == 1 else opts.fsl.k_query[i]

            opts.logger('\t\tcreating batch_ids in [{}] mode, '
                        'im_num:{:d}, {:d}-way, {:d}-shot, {:d}-query, im_resize:{:d}'.format(
                            phase.upper(), images.shape[0],
                            opts.fsl.n_way[i], curr_shot, curr_query, opts.data.im_size))
            train_data.append(tierImagenet(
                n_way=opts.fsl.n_way[i], k_shot=curr_shot, k_query=curr_query,
                resize=opts.data.im_size, augment=opts.data.augment,
                images=images, labels=labels, split=phase
            ))
    else:
        raise NameError('Unknown dataset ({})!'.format(opts.dataset.name))

    # turn data_loader into db
    train_db = [
        DataLoader(x, opts.train.batch_sz, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        for x in train_data
    ]
    val_db = [
        DataLoader(x, opts.test.batch_sz, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        for x in val_data
    ]

    # update opts
    opts.ctrl.total_iter_train = [len(each_db) for each_db in train_db] if not opts.ctrl.eager \
        else [100 for _ in train_db]
    opts.ctrl.total_iter_val = [len(each_db) for each_db in val_db] if not opts.ctrl.eager \
        else [50 for _ in val_db]

    return train_db, val_db, test_db, trainval_db
