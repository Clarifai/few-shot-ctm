import time
import argparse
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, StepLR

from core.model import CTMNet
from dataset.data_loader import data_loader
from tools.general_utils import *
from tools.visualize import Visualizer
from core.workflow import *
from core.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eid', type=int, default=-1)
    parser.add_argument('--gpu_id', type=int, nargs='+', default=0)
    parser.add_argument('--yaml_file', type=str, default='configs/demo/mini/20way_1shot.yaml')
    outside_opts = parser.parse_args()
    if isinstance(outside_opts.gpu_id, int):
        outside_opts.gpu_id = [outside_opts.gpu_id]  # int -> list

    config = {}
    config['options'] = {
        'ctrl.yaml_file': outside_opts.yaml_file,
        'ctrl.gpu_id': outside_opts.gpu_id
    }
    opts = Config(config['options']['ctrl.yaml_file'], config['options'])
    opts.setup()

    # DATA
    meta_test = None
    train_db_list, val_db_list, _, _ = data_loader(opts)

    # MODEL
    # NOTE: we use cpu mode for demo; change to gpu for experiments
    net = CTMNet(opts).to(opts.ctrl.device)

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
    if opts.model.structure == 'original':
        # ignore previous setting
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        opts.train.lr_policy = 'step'
        opts.train.step_size = 100 if not opts.data.use_ori_relation else 3
        opts.train.lr_scheduler = [-1]
        opts.train.lr = 0.001
        opts.train.lr_gamma = 0.5
        opts.train.weight_decay = .0

    # VISUALIZE
    if opts.misc.vis.use:
        if opts.misc.vis.method == 'tensorboard':
            NotImplementedError()
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
    opts.logger('CTM Pipeline starts now !!! (cpu demo purpose)')
    show_str = '[TRAIN FROM SCRATCH] LOG' if not opts.io.resume else '[RESUME] LOG'
    opts.logger('{}\n'.format(show_str))

    total_ep = opts.train.nep
    if opts.ctrl.start_epoch > 0 or opts.ctrl.start_iter > 0:
        assert opts.io.resume
        RESUME = True
    else:
        RESUME = False

    for epoch in range(opts.ctrl.start_epoch, total_ep):

        if epoch > opts.ctrl.start_epoch and opts.data.change_on_every_ep:
            opts.logger('')
            opts.logger('Changing a new set of data at new epoch ...')
            train_db_list, val_db_list, _, _ = data_loader(opts)

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
        curr_shot = opts.fsl.k_shot[0]
        curr_query = opts.fsl.k_query[0]                # only for display (for evolutionary train)
        train_db = train_db_list[0]
        val_db = val_db_list[0]
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

            support_x, support_y, query_x, query_y = process_input(batch, opts, mode='train')
            loss, _ = net.forward_CTM(support_x, support_y, query_x, query_y, True)
            loss = loss.mean(0)
            vis_loss = loss.data.cpu().numpy()

            vis_loss *= opts.train.total_loss_fac
            loss *= opts.train.total_loss_fac

            if len(loss) > 1:
                total_loss = loss[0]
            else:
                total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            if opts.train.clip_grad:
                # doesn't affect that much
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            iter_time = (time.time() - step_t)
            left_time = compute_left_time(iter_time, epoch, total_ep, step, total_iter)

            # SHOW TRAIN LOSS
            if step % opts.io.iter_vis_loss == 0 or step == total_iter - 1:
                opts.logger(opts.io.loss_vis_str.format(epoch, total_ep, step, total_iter, total_loss.item()))
                # time
                if step % 1000*opts.io.iter_vis_loss == 0 or step == total_iter - 1:
                    opts.logger(opts.io.time_vis_str.format(left_time[0], left_time[1], left_time[2]))

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
                    'curr_shot': curr_shot,
                    'curr_query': curr_query,
                    'best_accuracy': best_accuracy,
                    'last_epoch': last_epoch,
                    'last_iter':  last_iter,
                    'new_lr': new_lr,
                    'train_db': train_db,
                    'total_iter': total_iter,
                    'optimizer': optimizer,
                    'meta_test': meta_test
                }
                try:
                    stats = run_test(opts, val_db, net, vis, **arguments)
                except RuntimeError:
                    vis.show_dynamic_info(phase='error')
                if sum(stats) != -1:
                    best_accuracy, last_epoch, last_iter = stats[0], stats[1], stats[2]
            # DONE with validation process

    opts.logger('')
    opts.logger('Training done! check your work using:')
    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        vis.show_dynamic_info(phase='train_finish')
        if not opts.ctrl.eager:
            opts.logger('visdom state saved!')
            vis.save()


if __name__ == '__main__':
    main()
