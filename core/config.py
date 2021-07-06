from tools.general_utils import *


class Config(object):

    # DEFAULT VALUES
    dataset = AttrDict()
    dataset.name = 'mini-imagenet'

    data = AttrDict()
    data.im_size = 84
    # see specific files for each set under dataset
    # dataset/mini_imagenet.py
    data.augment = '0'
    data.use_batch_sampler = False
    data.change_on_every_ep = False
    data.use_ori_relation = False

    model = AttrDict()
    # 19, 52, 34, shallow
    model.structure = 'resnet40'
    model.resnet_pretrain = False
    # this is the default. other choice: 'simple'
    # if relation network is used, which module to choose
    # (resnet_block or simple)
    model.relation_net = 'res_block'

    # ==============
    fsl = AttrDict()                # few-shot learning
    fsl.ctm = True
    fsl.n_way = [5]
    fsl.k_shot = [5]
    fsl.k_query = [5]
    fsl.epoch_schedule = [10, 30, 40]
    fsl.triplet = False
    fsl.swap = False
    fsl.swap_num = 0
    fsl.meta_learn = 'nope'
    fsl.CE_loss = None
    fsl.hier = None

    tri = AttrDict()                # few-shot learning
    mlearn = AttrDict()                # few-shot learning

    # ==============
    ctmnet = AttrDict()
    ctmnet.CE_use_relation = False
    ctmnet.baseline_manner = ''     # 'sample_wise', 'sum', 'no_reshaper'
    ctmnet.dnet = True
    ctmnet.pred_source = 'score'
    ctmnet.dnet_out_c = 40            # shallow output channel is 64
    ctmnet.dnet_supp_manner = '1'
    ctmnet.use_discri_loss = False

    # ==============
    io = AttrDict()
    io.root = 'output'
    io.exp_name = 'default'

    io.output_folder = ''
    io.model_file = ''
    io.log_file = ''
    io.logger = None                # class object
    io.iter_vis_loss = 5
    io.iter_do_val = 1000
    io.loss_vis_str = ''
    io.resume = False
    io.previous_acc = 0

    # ==============
    ctrl = AttrDict()
    # fasten training process (vis loss, total_iter per epoch, etc)
    ctrl.eager = 0
    ctrl.method = 'relation'
    ctrl.yaml_file = ''
    # gpu_id is just for visualization purpose
    ctrl.gpu_id = [0]

    # ignore the resume file if set True
    ctrl.train_from_scratch = False
    ctrl.device = 'cpu'
    ctrl.multi_gpu = False
    # how many iteration we want for each epoch and each test evaluation
    ctrl.total_iter_train = -1
    ctrl.total_iter_val = -1
    ctrl.start_epoch = 0
    ctrl.start_iter = 0
    # pretrain resnet on mini-imagent or tier-imagenet
    ctrl.pretrain = False

    # ==============
    train = AttrDict()
    train.batch_sz = 1
    train.nep = 500
    train.lr_policy = 'multi_step'
    train.lr_scheduler = [20, 30]
    train.lr = 0.001
    train.lr_gamma = 0.5
    train.weight_decay = .0005
    train.total_loss_fac = 1.
    train.optim = 'adam'
    train.momentum = 0.9        # only for sgd method
    train.clip_grad = False

    # ==============
    test = AttrDict()
    test.batch_sz = -1
    # determine the number of query; 'same_as_train'
    test.manner = 'standard'
    test.ep_num = 600
    test.query_num = 15
    test.compute_train_acc = True
    test.do_after_ep = -1

    # ==============
    misc = AttrDict()
    misc.manual_seed = -1
    misc.vis = AttrDict()
    misc.vis.use = False

    def __init__(self, yaml_file, options=None):

        if yaml_file:
            merge_cfg_from_file(yaml_file, self)

        if options is not None:
            outside = []
            for k, v in options.items():
                outside.append(k)
                outside.append(v)
            merge_cfg_from_list(outside, self)

    def setup(self):
        # set seed
        if self.misc.manual_seed == -1:
            self.misc.manual_seed = np.random.randint(0, 10000)
        np.random.seed(self.misc.manual_seed)
        torch.manual_seed(self.misc.manual_seed)
        torch.cuda.manual_seed(self.misc.manual_seed)

        # set up for re-implementing the original relation network
        if self.ctrl.method == 'all_new_1' or self.ctrl.method == 'all_new_3':
            self.model.structure = 'original'
        if self.ctrl.method == 'all_new_2' or self.ctrl.method == 'all_new_3':
            self.data.use_ori_relation = True

        # set test batch size
        if self.test.batch_sz == -1:
            self.test.batch_sz = self.train.batch_sz

        # eager mode
        if self.ctrl.eager:
            self.io.iter_vis_loss, self.io.iter_do_val = 5, 102
            self.train.lr_scheduler = [5, 7]
            self.train.nep = 10
            # self.train.lr_scheduler = [10]
            # self.train.nep = 15
            # self.test.do_after_ep = 15
            self.test.do_after_ep = -1
        else:
            if self.test.do_after_ep == -1:
                self.test.do_after_ep = self.train.lr_scheduler[0] - 10
        if self.data.use_ori_relation:
            # override
            self.test.do_after_ep = -1

        # set up opt.log_file
        # output_folder: output/METHOD/DATASET/exp_name
        self.io.output_folder = os.path.join(
            self.io.root, self.ctrl.method, self.dataset.name, self.io.exp_name)
        if not os.path.exists(self.io.output_folder):
            os.makedirs(self.io.output_folder)
        self.io.log_file = os.path.join(self.io.output_folder, 'training_dynamic.txt')

        # set up logger
        self.io.logger = Logger(log_file=self.io.log_file)
        # for god's sake, since Logger is using too often, we will have legacy here
        self.logger = self.io.logger
        display = 'Writing params into log: {}\n'
        self.io.logger(display.format(self.io.log_file), init=True)

        # set opt.model_file (based on opt.output_folder)
        if self.ctrl.pretrain:
            self.io.model_file = os.path.join(
                self.io.output_folder, '{:s}_{:s}_best_pretrain_model.pt'.format(
                    self.dataset.name.replace('-', '_'), self.model.structure))
        else:
            self.io.model_file = os.path.join(
                self.io.output_folder, 'best_model_{:d}_way_{:d}_shot.pt'.format(
                    self.fsl.n_way[-1], self.fsl.k_shot[-1]))
        self.io.logger('Model file is saved as {}\n'.format(self.io.model_file))

        # visualization
        self.io.loss_vis_str = ' [ep {:04d} ({})/ iter {:06d} ({})] loss: {:.4f}'
        self.io.time_vis_str = ' \tEstimated left time: {:d} days, {:.4f} hours;\tTotal time taken: {:.2f} days'
        self._sanity_check()

        # set opt.multi_gpu and opt.device
        multi_gpu = True if len(self.ctrl.gpu_id) > 1 else False
        self.logger('gpu_ids: {}\n'.format(self.ctrl.gpu_id))
        self.ctrl.multi_gpu = multi_gpu

    def _sanity_check(self):

        if not self.ctrl.pretrain:
            assert self.test.batch_sz == 1
            assert self.train.batch_sz == 1

        if len(self.fsl.n_way) > 1:
            self.fsl.evolution = True
            assert len(self.fsl.k_shot) == 1 or len(self.fsl.k_shot) == len(self.fsl.n_way)
            assert len(self.fsl.k_query) == 1 or len(self.fsl.k_query) == len(self.fsl.n_way)
            assert len(self.fsl.n_way) == len(self.fsl.epoch_schedule) + 1, \
                'len(n_way) should be equal to len(epoch_schedule)+1; actual is ({}) == ({})+1'.format(
                    len(self.fsl.n_way), len(self.fsl.epoch_schedule)
                )
        else:
            self.fsl.evolution = False
            assert len(self.fsl.k_shot) == 1 and len(self.fsl.k_query) == 1
            del self.fsl['epoch_schedule']

        if not self.misc.vis.use:
            del self.misc['vis']
            self.misc.vis = AttrDict()
            self.misc.vis.use = False

        if self.train.lr_policy != 'multi_step':
            del self.train['lr_scheduler']

        if not self.fsl.triplet:
            del self.tri
        if self.fsl.triplet and self.tri.use_tri_only:
            # self.tri.test_source = 'n/a'
            del self.tri['loss_fac']
            del self.tri['test_source']
        if self.fsl.triplet and self.tri.method == 'ratio':
            del self.tri['margin']

        if self.test.manner == 'same_as_train':
            del self.test['ep_num']
            del self.test['query_num']
        if self.train.optim == 'adam':
            del self.train['momentum']
        if not self.fsl.swap:
            del self.fsl['swap_num']
        else:
            assert self.fsl.swap_num > 1, 'swap number should be over 1'

        if self.ctrl.method == 'meta_learn':
            assert self.fsl.meta_learn in ['reptile', 'maml', 'meta_lstm']
            # if self.fsl.meta_learn == 'reptile':
            #     assert self.train.batch_sz == 1
            self.mlearn.outer_lr = self.mlearn.lr_fac * self.train.lr
        else:
            assert self.fsl.meta_learn == 'nope'
            del self.mlearn

        if self.data.use_ori_relation:
            assert not self.data.change_on_every_ep
            assert not self.data.use_batch_sampler
        if not self.fsl.ctm:
            del self.ctmnet
        else:
            # use new pipeline
            del self.fsl['CE_loss']
            del self.fsl['hier']
            del self.fsl['triplet']
            del self.fsl['meta_learn']
            if self.ctmnet.dnet:
                del self.ctmnet['baseline_manner']
            else:
                del self.ctmnet['dnet_supp_manner']
                del self.ctmnet['use_discri_loss']
                del self.ctmnet['discri_random_target']
                del self.ctmnet['discri_test_update']
                del self.ctmnet['discri_random_weight']
                del self.ctmnet['dnet_mp_mean']
                del self.ctmnet['dnet_delete_mp']

        if self.ctrl.pretrain:
            try:
                del self.fsl
                del self.ctmnet
                del self.tri
                del self.mlearn
            except:
                pass

    def _print_attr_dict(self, k, v, indent):
        self.logger('{:s}{:s}:'.format(indent, k), quiet_ter=True)
        for _k, _v in sorted(v.items()):
            if isinstance(_v, AttrDict):
                self._print_attr_dict(_k, _v, indent=indent+'\t')
            elif isinstance(_v, Logger):
                self.logger('{:s}\t{:s}:\t\t\t{}'.format(indent, _k, 'Class object not shown here'))
            else:
                self.logger('{:s}\t{:s}:\t\t\t{}'.format(indent, _k, _v))

    def print_args(self):
        # ALL in ALL
        self.logger('CONFIGURATION BELOW')
        temp = self.__dict__
        for k in sorted(temp):
            if isinstance(temp[k], AttrDict):
                self._print_attr_dict(k, temp[k], indent='\t')
            elif isinstance(temp[k], bool):
                self.logger('\t{:s}:\t\t{}'.format(k, temp[k]), quiet_ter=True)
            else:
                self.logger('\t{:s}:\t\t{}'.format(k, 'Class object not shown here'), quiet_ter=True)
        self.logger('\n')
