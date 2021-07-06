import time
import traceback
import sys
from torch.nn import functional as F
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, StepLR

from dataset.data_loader_pretrain import data_loader_pretrain
from tools.general_utils import *
from tools.visualize import Visualizer
from workflow import *
from config import Config
from core.model import feat_extract


def main(config=None):

    opts = config
    opts.setup()

    train_db, val_db = data_loader_pretrain(opts)
    # MODEL
    model_kwargs = {
        'pretrained': opts.model.resnet_pretrain,
        'structure': opts.model.structure,
        'in_c': 3,
        'num_classes': train_db.dataset.cls_num,
        'opts': opts,
        'include_head': True,
    }
    net = feat_extract(**model_kwargs)
    net = net.to(opts.ctrl.device)

    net_summary, param_num = model_summarize(net)
    opts.logger('Model size: param num # {:f} Mb'.format(param_num))
    opts.model.param_size = param_num

    resume_model(net, opts)
    if opts.ctrl.multi_gpu:
        opts.logger('Wrapping network into multi-gpu mode ...')
        net = torch.nn.DataParallel(net)

    # OPTIM AND LR SCHEDULE
    if opts.train.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay)
    elif opts.train.optim == 'sgd':
        optimizer = optim.SGD(
            net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay, momentum=opts.train.momentum)
    elif opts.train.optim == 'rmsprop':
        optimizer = optim.RMSprop(
            net.parameters(), lr=opts.train.lr, weight_decay=opts.train.weight_decay, momentum=opts.train.momentum,
            alpha=0.9, centered=True)
    if opts.train.lr_policy == 'multi_step':
        scheduler = MultiStepLR(optimizer, milestones=opts.train.lr_scheduler, gamma=opts.train.lr_gamma)
    elif opts.train.lr_policy == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=opts.train.lr_gamma)

    # VISUALIZE
    vis = None
    if opts.misc.vis.use:
        if opts.misc.vis.method == 'tensorboard':
            NotImplementedError()
            # from tensorboardX import SummaryWriter
            # tb = SummaryWriter(opts.tb_folder, str(datetime.now()))
        elif opts.misc.vis.method == 'visdom':
            if opts.io.resume:
                try:
                    vis = Visualizer(opts, net.previous_loss_data)
                except:
                    vis = Visualizer(opts, net.module.previous_loss_data)
            else:
                vis = Visualizer(opts)

    if not opts.ctrl.eager:
        opts.print_args()
        opts.logger(net)
    else:
        opts.logger('config file is {:s}'.format(opts.ctrl.yaml_file))
        opts.logger('configs not shown here in eager mode ...')
        opts.logger(net)

    # ###############################################
    # ################## PIPELINE ###################
    best_accuracy = opts.io.previous_acc
    RESET_BEST_ACC = True   # for evolutionary train
    last_epoch, last_iter = opts.io.saved_epoch, opts.io.saved_iter
    opts.logger('Pipeline starts now !!!')

    total_ep = opts.train.nep
    if opts.ctrl.start_epoch > 0 or opts.ctrl.start_iter > 0:
        assert opts.io.resume
        RESUME = True
    else:
        RESUME = False
    VERY_FIRST_TIME = True

    for epoch in range(opts.ctrl.start_epoch, total_ep):

        # adjust learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch)
        new_lr = optimizer.param_groups[0]['lr']
        if epoch == opts.ctrl.start_epoch:
            opts.logger('Start lr is {:.8f}, at epoch {}\n'.format(old_lr, epoch))
        if new_lr != old_lr:
            opts.logger('LR changes from {:.8f} to {:.8f} at epoch {:d}\n'.format(old_lr, new_lr, epoch))

        # select proper train_db (legacy reason)
        which_ind = 0
        total_iter = opts.ctrl.total_iter_train[0]
        eval_length = opts.ctrl.total_iter_val[0]

        for step, batch in enumerate(train_db):

            step_t = time.time()
            if RESUME:
                if step < opts.ctrl.start_iter:
                    continue
                else:
                    RESUME = False

            if step >= total_iter:
                break
            x, y = batch[0].to(opts.ctrl.device), batch[1].to(opts.ctrl.device)
            prediction = net(x)
            loss = F.cross_entropy(prediction, y.squeeze(1))
            loss = loss.mean(0)
            vis_loss = loss.data.cpu().numpy()

            total_loss = loss
            optimizer.zero_grad()
            total_loss.backward()
            if opts.train.clip_grad:
                # doesn't affect that much
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            iter_time = (time.time() - step_t)
            left_time = compute_left_time(iter_time, epoch, total_ep, step, total_iter)
            info = {
                'curr_ep': epoch,
                'curr_iter': step,
                'total_ep': total_ep,
                'total_iter': total_iter,
                'loss': vis_loss,
                'left_time': left_time,
                'lr': new_lr,
                'iter_time': iter_time
            }
            # SHOW TRAIN LOSS
            if step % opts.io.iter_vis_loss == 0 or step == total_iter - 1 or VERY_FIRST_TIME:
                VERY_FIRST_TIME = False
                # loss
                opts.logger(opts.io.loss_vis_str.format(epoch, total_ep, step, total_iter, total_loss.item()))
                # time
                if step % 1000*opts.io.iter_vis_loss == 0 or step == total_iter - 1:
                    opts.logger(opts.io.time_vis_str.format(left_time[0], left_time[1], left_time[2]))
                # visdom
                if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
                    # tb.add_scalar('loss', loss.item())
                    vis.plot_loss(**info)
                    vis.show_dynamic_info(**info)

            # VALIDATION and SAVE BEST MODEL
            if epoch > opts.test.do_after_ep and \
                    ((step % opts.io.iter_do_val == 0 and step > 0) or step == total_iter - 1):

                # execute once only
                if RESET_BEST_ACC and opts.fsl.evolution and epoch >= opts.fsl.epoch_schedule[-1]:
                    best_accuracy, last_epoch, last_iter = -1.0, -1, -1
                    RESET_BEST_ACC = False

                arguments = {
                    'step': step,
                    'epoch': epoch,
                    'eval_length': eval_length,
                    'which_ind': which_ind,
                    'best_accuracy': best_accuracy,
                    'last_epoch': last_epoch,
                    'last_iter':  last_iter,
                    'new_lr': new_lr,
                    'train_db': train_db,
                    'total_iter': total_iter,
                }
                try:
                    stats = run_test_pretrain(opts, val_db, net, vis, **arguments)
                except RuntimeError:
                    if vis:
                      vis.show_dynamic_info(phase='error')
                    traceback.print_exc()
                if sum(stats) != -1:
                    best_accuracy, last_epoch, last_iter = stats[0], stats[1], stats[2]
            # DONE with validation process

    opts.logger('')
    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        vis.show_dynamic_info(phase='train_finish')
        if not opts.ctrl.eager:
            opts.logger('visdom state saved!')
            vis.save()


if __name__ == '__main__':
    main(Config(sys.argv[1]))
