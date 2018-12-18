import os
import torch
import numpy as np
import pickle as pkl
import cv2
import tempfile
from cStringIO import StringIO
import math
import yaml
import copy
from ast import literal_eval
from torch.nn.modules.module import _addindent


def model_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    params_num = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = model_summarize(module)
        else:
            modstr = module.__repr__()

        if isinstance(modstr, str):
            modstr = _addindent(modstr, 2)
        elif isinstance(modstr, tuple):
            modstr = _addindent(modstr[0], 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        # rest = b if b > 0 else params
        # params_num = params_num + rest
        params_num += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr, params_num * 4. / (1024**2)


def get_grad(mdoel):
    NotImplementedError()
    # TODO


def resume_model(model, opt):
    if opt.ctrl.train_from_scratch:
        opt.logger('train from scratch; ignore previous checkpoints')
        return None

    elif os.path.exists(opt.io.model_file):
        checkpoints = torch.load(opt.io.model_file)
        opt.io.resume = True

    if opt.io.resume:
        opt.logger('Loading previous best checkpoint [{}] ...'.format(opt.io.model_file))
        model.load_state_dict(checkpoints['state_dict'])
        _last_epoch = checkpoints['epoch']
        _last_lr = checkpoints['lr']
        _last_iter = checkpoints['iter']
        opt.io.previous_acc = checkpoints['val_acc']
        opt.logger('\tthis checkpoint is at epoch {}, iter {}, accuracy is {:.4f}\n'.format(
            _last_epoch, _last_iter, opt.io.previous_acc))

        if opt.fsl.evolution:
            which_ind = sum(_last_epoch >= np.array(opt.fsl.epoch_schedule))
        else:
            which_ind = 0
        if _last_iter == opt.ctrl.total_iter_train[which_ind] - 1:
            opt.ctrl.start_epoch = _last_epoch + 1
            opt.ctrl.start_iter = 0
        else:
            opt.ctrl.start_epoch = _last_epoch
            opt.ctrl.start_iter = _last_iter + 1

        if opt.misc.vis.use and opt.misc.vis.method == 'visdom':
            model.previous_loss_data = checkpoints['loss_data']

        opt.io.saved_epoch = _last_epoch
        opt.io.saved_iter = _last_iter
    else:
        opt.io.saved_epoch = 0
        opt.io.saved_iter = 0

    opt.logger('start_epoch is {}, start_iter is {}'.format(
        opt.ctrl.start_epoch, opt.ctrl.start_iter))


def print_args(opt):
    # ALL in ALL
    opt.logger('CONFIGURATION BELOW')
    temp = opt.__dict__
    for k in sorted(temp):
        opt.logger('\t{:s}:\t\t{}'.format(k, temp[k]), quiet_ter=True)


def remove(file_name):
    try:
        os.remove(file_name)
    except:
        pass


# new interface
class Logger(object):
    def __init__(self, log_file):
        self.file = log_file   # file always exists

    def __call__(self, msg, init=False, quiet_ter=False, additional_file=None):
        if not quiet_ter:
            print(msg)

        if init:
            remove(self.file)
            if additional_file is not None:
                remove(additional_file)

        with open(self.file, 'a') as log_file:
            log_file.write('%s\n' % msg)
        if additional_file is not None:
            with open(additional_file, 'a') as addition_log:
                addition_log.write('%s\n' % msg)


# old interface (deprecated)
def print_log(msg, file=None,
              init=False, additional_file=None,
              quiet_termi=False):

    if not quiet_termi:
        print(msg)
    if file is None:
        pass

    if file is not None:
        if init:
            remove(file)
        with open(file, 'a') as log_file:
            log_file.write('%s\n' % msg)

        if additional_file is not None:
            # TODO (low): a little buggy here: no removal of previous additional_file
            with open(additional_file, 'a') as addition_log:
                addition_log.write('%s\n' % msg)


def im_map_back(im, std, mean):
    im = im * torch.FloatTensor(list(std)).view(1, 3, 1, 1) + \
         torch.FloatTensor(list(mean)).view(1, 3, 1, 1)
    return im


# NOT USED
def show_result(opts, support_x, support_y, query_x, query_y, query_pred,
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """show result of one dimension (cvpr 2018 relation paper)"""
    n_way = opts.n_way
    k_shot = opts.k_shot
    n_query_per_cls = opts.k_query
    batchsz = support_x.size(0)
    device = opts.device

    # randomly select one batch
    batchidx = np.random.randint(batchsz)
    max_width = (k_shot + n_query_per_cls * 2 + 4)   # hyli: what's this?
    # de-normalize
    img_support = support_x[batchidx].clone()
    img_query = query_x[batchidx].clone()

    img_support = im_map_back(img_support, std, mean)
    img_query = im_map_back(img_query, std, mean)

    # TODO (irrelevant): no idea what's going on here; check later
    label = support_y[batchidx]                                             # [setsz]
    label, indices = torch.sort(label, dim=0)
    img_support = torch.index_select(img_support, dim=0, index=indices)     # [setsz, c, h, w]
    all_img = torch.zeros(max_width*n_way, *img_support[0].size())          # [max_width*n_way, c, h, w]

    for row in range(n_way):  # for each row
        # [0, k_shot)
        for pos in range(k_shot):  # copy the first k_shot
            all_img[row * max_width + pos] = img_support[row * k_shot + pos].data

        # now set the pred imgs
        # [k_shot+1, max_width - n_query_per_cls -1]
        pos = k_shot + 1  # pointer to empty buff
        for idx, img in enumerate(img_query):
            # search all imgs in pred that match current row id: label[row*k_shot]
            if torch.equal(query_pred[batchidx][idx], label[row * k_shot]):  # if pred it match current id
                if pos == max_width - n_query_per_cls:  # overwrite the last column
                    pos -= 1
                all_img[row * max_width + pos] = img.data  # copy img
                pos += 1

        # set the last several column as the right img
        #  [max_width - n_query_per_cls, max_width)
        pos = max_width - n_query_per_cls
        for idx, img in enumerate(img_query):  # search all imgs in pred that match current row id: label[row*k_shot]
            if torch.equal(query_y[batchidx][idx], label[row * k_shot]):  # if query_y id match current id
                if pos == max_width:  # overwrite the last column
                    pos -= 1
                all_img[row * max_width + pos] = img.data  # copy img
                pos += 1

    print('label for support:', label.data.cpu().numpy().tolist())
    print('label for query  :', query_y.data[batchidx].cpu().numpy())
    print('label for pred   :', query_pred.data[batchidx].cpu().numpy())

    return all_img, max_width


# the following functions are from tier-imagenet
# def compress(path, output):
#     with np.load(path, mmap_mode="r") as data:
#         images = data["images"]
#         array = []
#         for ii in tqdm(six.moves.xrange(images.shape[0]), desc='compress'):
#             im = images[ii]
#             im_str = cv2.imencode('.png', im)[1]
#             array.append(im_str)
#     with open(output, 'wb') as f:
#         pkl.dump(array, f, protocol=pkl.HIGHEST_PROTOCOL)


def decompress(path, output):
    with open(output, 'rb') as f:
        array = pkl.load(f, encoding='latin1')

    print_log('\t\ndecompressing the raw data: {} to {} ...'.format(output, path))
    images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
    for ii, item in enumerate(array):
        im = cv2.imdecode(item, 1)
        images[ii] = im
    np.savez(path, images=images)


def compute_left_time(iter_avg, curr_ep, total_ep, curr_iter, total_iter):

    total_time = ((total_iter - curr_iter) + (total_ep - curr_ep)*total_iter) * iter_avg
    days = int(math.floor(total_time / (3600.0*24)))  # int type
    hrs = (total_time - days*3600.0*24) / 3600

    total_time_taken = (curr_iter + curr_ep*total_iter)*iter_avg
    days_taken = total_time_taken / (3600.0*24)   # float type
    return days, hrs, days_taken


def _cls2dict(config):
    output = AttrDict()
    for a in dir(config):
        value = getattr(config, a)
        if not a.startswith("__") and not callable(value):
            assert isinstance(value, AttrDict)
            output[a] = value
    return output


def _dict2cls(_config, config):
    for a in dir(config):
        if not a.startswith("__") and not callable(getattr(config, a)):
            setattr(config, a, _config[a])


def merge_cfg_from_file(cfg_filename, config):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _config = _cls2dict(config)
    _merge_a_into_b(yaml_cfg, _config)
    _dict2cls(_config, config)


def merge_cfg_from_list(cfg_list, config):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    _config = _cls2dict(config)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = _config
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value
    _dict2cls(_config, config)


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, basestring):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, basestring):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]
