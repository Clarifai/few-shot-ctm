import torch
from copy import deepcopy
import archive.task_generator_test_revise as task_gen


def process_input(batch, opts, mode='train'):
    if opts.data.use_batch_sampler:
        all_data, label = batch[0], batch[1]
        # all_data: 50, 1, 30, 30; label: 50
        _c, _w = all_data.size(1), all_data.size(2)

        n_way, k_shot = opts.fsl.n_way[0], opts.fsl.k_shot[0]
        support_x = torch.zeros(n_way * k_shot, _c, _w, _w).to(opts.ctrl.device)
        support_y = torch.zeros(n_way * k_shot).to(opts.ctrl.device)
        if (mode == 'test' or mode == 'val') and opts.test.manner == 'standard':
            k_query = opts.test.query_num
        else:
            k_query = opts.fsl.k_query[0]
        query_x = torch.zeros(n_way * k_query, _c, _w, _w).to(opts.ctrl.device)
        query_y = torch.zeros(n_way * k_query).to(opts.ctrl.device)

        cls = torch.unique(label)
        for i, curr_cls in enumerate(cls):

            support_y[i*k_shot:i*k_shot + k_shot] = curr_cls
            query_y[i*k_query:i*k_query + k_query] = curr_cls

            curr_data = all_data[curr_cls == label]
            support_x[i*k_shot:i*k_shot + k_shot] = curr_data[:k_shot]
            query_x[i*k_query:i*k_query + k_query] = curr_data[k_shot:]

        support_x, support_y, query_x, query_y = \
            support_x.unsqueeze(0), support_y.unsqueeze(0), \
            query_x.unsqueeze(0), query_y.unsqueeze(0)

    else:
        # support_x: support_sz, 3, 84, 84
        support_x, support_y, query_x, query_y = \
            batch[0].to(opts.ctrl.device), batch[1].to(opts.ctrl.device), \
            batch[2].to(opts.ctrl.device), batch[3].to(opts.ctrl.device)

    return support_x, support_y, query_x, query_y


def test_model(net, input_db, eval_length, opts, which_ind, curr_shot, optimizer=None, meta_test=None):
    """
    optimizer is for meta-test only
    meta_test is for using the dataloader in the original relation codebase. Not the same meaning as "meta_learn"
    """
    total_correct, total_num, display_onebatch = \
        torch.zeros(1).to('cuda'), torch.zeros(1).to('cuda'), False

    if opts.ctrl.method == 'meta_learn':

        adapt_num = opts.mlearn.adapt_num if opts.mlearn.adapt_num_test == -1 else opts.mlearn.adapt_num_test
        weights_before = deepcopy(net.state_dict())
        # with torch.no_grad():
        for j, batch_test in enumerate(input_db):

            if j >= eval_length:
                break
            support_x, support_y, query_x, query_y = process_input(batch_test, opts, mode='test')

            for i in range(adapt_num):
                # shape: gpu_num x loss_num
                loss = net(support_x, support_y, None, None)
                loss = loss.mean(0)
                loss *= opts.train.total_loss_fac

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            correct = net(None, None, query_x, query_y, False)
            total_correct += correct.sum().float()  # multi-gpu support
            total_num += query_y.size(0) * query_y.size(1)

        net.load_state_dict(weights_before)  # restore from snapshot

        accuracy = total_correct / total_num  # due to python 2, it's converted to int!
        accuracy = accuracy.item()

    elif opts.fsl.ot and hasattr(opts.otnet, 'use_discri_loss') and \
            opts.otnet.use_discri_loss and opts.otnet.discri_test_update:

        for j, batch_test in enumerate(input_db):

            if j >= eval_length:
                break

            support_x, support_y, query_x, query_y = process_input(batch_test, opts, mode='test')
            _, correct = net.forward_OT(support_x, support_y, query_x, query_y, False, optimizer=optimizer)
            total_correct += correct.sum().float()  # multi-gpu support
            total_num += query_y.numel()

        accuracy = total_correct / total_num  # due to python 2, it's converted to int!
        accuracy = accuracy.item()

    else:
        net.eval()
        with torch.no_grad():
            for j, batch_test in enumerate(input_db):

                if j >= eval_length:
                    break
                if opts.data.use_ori_relation:
                    task = task_gen.MiniImagenetTask(
                        meta_test, num_classes=opts.fsl.n_way[0], train_num=curr_shot, test_num=15)
                    support_db = task_gen.get_data_loader(task, num_per_class=curr_shot, split='support', shuffle=False)
                    query_db = task_gen.get_data_loader(task, num_per_class=15, split='query', shuffle=False)
                    support_x, support_y = support_db.__iter__().next()
                    query_x, query_y = query_db.__iter__().next()
                    support_x, support_y, query_x, query_y = \
                        support_x.unsqueeze(0).to(opts.ctrl.device), support_y.unsqueeze(0).to(opts.ctrl.device), \
                        query_x.unsqueeze(0).to(opts.ctrl.device), query_y.unsqueeze(0).to(opts.ctrl.device)
                else:
                    support_x, support_y, query_x, query_y = process_input(batch_test, opts, mode='test')

                if opts.fsl.ot:
                    _, correct = net.forward_OT(support_x, support_y, query_x, query_y, False)
                else:
                    if opts.model.structure == 'original':
                        support_x, support_y, query_x, query_y = \
                            support_x.squeeze(0), support_y.squeeze(0), query_x.squeeze(0), query_y.squeeze(0)
                        _, correct = net(support_x, support_y, query_x, query_y, False)
                    else:
                        _, correct = net(support_x, support_y, query_x, query_y, False,
                                         opts.fsl.n_way[which_ind], curr_shot)

                total_correct += correct.sum().float()  # multi-gpu support
                total_num += query_y.numel()

        accuracy = total_correct / total_num  # due to python 2, it's converted to int!
        accuracy = accuracy.item()
        # if opts.misc.vis.method == 'tensorboard':
        #     NotImplementedError()
        #     # tb.add_scalar('accuracy', accuracy)
        net.train()
    return accuracy


def test_model_pretrain(net, input_db, eval_length, opts):

    total_correct, total_num, display_onebatch = \
        torch.zeros(1).to('cuda'), torch.zeros(1).to('cuda'), False

    net.eval()
    with torch.no_grad():
        for j, batch_test in enumerate(input_db):

            if j >= eval_length:
                break

            x, y = batch_test[0].to(opts.ctrl.device), batch_test[1].to(opts.ctrl.device)
            predict = net(x).argmax(dim=1, keepdim=True)
            correct = torch.eq(predict, y)

            # compute correct
            total_correct += correct.sum().float()  # multi-gpu support
            total_num += predict.numel()

    accuracy = total_correct / total_num  # due to python 2, it's converted to int!
    accuracy = accuracy.item()
    net.train()
    return accuracy


def run_test(opts, val_db, net, vis, **args):
    step = args['step']
    epoch = args['epoch']
    eval_length = args['eval_length']
    which_ind = args['which_ind']
    curr_shot = args['curr_shot']
    curr_query = args['curr_query']           # only for display (evolutionary train)
    best_accuracy = args['best_accuracy']
    last_epoch = args['last_epoch']
    last_iter = args['last_iter']
    new_lr = args['new_lr']
    train_db = args['train_db']
    total_iter = args['total_iter']
    optimizer = args['optimizer']
    try:
        meta_test = args['meta_test']
    except KeyError:
        meta_test = None

    _tmp = '<br/><br/><b>TEST</b><br/>'
    _curr_str = '\tEvaluating at epoch {}, step {}, with eval_length {} ... (be patient)'.format(
        epoch, step, int(eval_length))
    opts.logger(_curr_str)
    _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'
    if opts.fsl.evolution:
        _curr_str = '\t---- n_way {}, k_shot {}, k_query {} ----'.format(
            opts.fsl.n_way[which_ind], curr_shot, curr_query)
        opts.logger(_curr_str)
        _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'

    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        info = {'msg': _tmp}
        vis.show_dynamic_info(phase='test', **info)
        vis._show_config()

    accuracy = test_model(net, val_db, eval_length, opts, which_ind, curr_shot, optimizer, meta_test)

    eqn = '>' if accuracy > best_accuracy else '<'
    if opts.fsl.evolution:
        _str = '(true)' if epoch >= opts.fsl.epoch_schedule[-1] else '(pseudo)'
    else:
        _str = ''
    _curr_str = '\t\tCurrent {:s} accuracy is {:.4f} {:s} ' \
                'previous best accuracy is {:.4f} (ep{}, iter{})'.format(
        _str, accuracy, eqn, best_accuracy, last_epoch, last_iter)
    opts.logger(_curr_str)
    _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'
    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        info = {'msg': _tmp}
        vis.show_dynamic_info(phase='test', **info)
        if not opts.ctrl.eager:
            vis.save()

    # Also test the train-accuracy at end of one epoch
    if opts.test.compute_train_acc and step == total_iter - 1 and not opts.ctrl.eager:
        _curr_str = '\tEvaluating training acc at epoch {}, step {}, length {} ... (be patient)'.format(
            epoch, step, len(train_db))
        opts.logger(_curr_str)
        _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'

        train_acc = test_model(net, train_db, len(train_db), opts, which_ind, curr_shot, optimizer, meta_test)

        _curr_str = '\t\tCurrent train_accuracy is {:.4f} at END of epoch {:d}'.format(
            train_acc, epoch)
        opts.logger(_curr_str)
        _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'
        opts.logger('')
        if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
            info = {'msg': _tmp}
            vis.show_dynamic_info(phase='test', **info)

    _tmp = _tmp.replace('<b>TEST</b>', 'Last test stats')
    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        info = {'msg': _tmp}
        vis.show_dynamic_info(phase='test_finish', **info)

    # SAVE MODEL
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        last_epoch = epoch
        last_iter = step
        model_weights = net.module.state_dict() if opts.ctrl.multi_gpu else net.state_dict()
        file_to_save = {
            'state_dict': model_weights,
            'lr': new_lr,
            'epoch': epoch,
            'iter': step,
            'val_acc': accuracy,
            'options': opts,
        }
        if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
            file_to_save.update({'loss_data': vis.loss_data})
            info = {
                'acc': accuracy,
                'epoch': epoch,
                'iter': step,
                'path': opts.io.model_file
            }
            vis.show_best_model(**info)

        torch.save(file_to_save, opts.io.model_file)
        opts.logger('\tBest model saved to: {}, at [epoch {} / iter {}]\n'.format(
            opts.io.model_file, epoch, step))

        return [best_accuracy, last_epoch, last_iter]
    else:
        return [-1]
    # DONE WITH SAVE MODEL


def run_test_pretrain(opts, val_db, net, vis, **args):
    step = args['step']
    epoch = args['epoch']
    eval_length = args['eval_length']
    best_accuracy = args['best_accuracy']
    last_epoch = args['last_epoch']
    last_iter = args['last_iter']
    new_lr = args['new_lr']
    train_db = args['train_db']
    total_iter = args['total_iter']

    _tmp = '<br/><br/><b>TEST</b><br/>'
    _curr_str = '\tEvaluating at epoch {}, step {}, with eval_length {} ... (be patient)'.format(
        epoch, step, int(eval_length))
    opts.logger(_curr_str)
    _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'

    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        info = {'msg': _tmp}
        vis.show_dynamic_info(phase='test', **info)
        vis._show_config()

    accuracy = test_model_pretrain(net, val_db, eval_length, opts)

    eqn = '>' if accuracy > best_accuracy else '<'
    _str = ''
    _curr_str = '\t\tCurrent {:s} accuracy is {:.4f} {:s} ' \
                'previous best accuracy is {:.4f} (ep{}, iter{})'.format(
        _str, accuracy, eqn, best_accuracy, last_epoch, last_iter)
    opts.logger(_curr_str)
    _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'
    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        info = {'msg': _tmp}
        vis.show_dynamic_info(phase='test', **info)
        if not opts.ctrl.eager:
            vis.save()

    # Also test the train-accuracy at end of one epoch
    if opts.test.compute_train_acc and step == total_iter - 1 and not opts.ctrl.eager:
        _curr_str = '\tEvaluating training acc at epoch {}, step {}, length {} ... (be patient)'.format(
            epoch, step, len(train_db))
        opts.logger(_curr_str)
        _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'

        train_acc = test_model_pretrain(net, train_db, len(train_db), opts)

        _curr_str = '\t\tCurrent train_accuracy is {:.4f} at END of epoch {:d}'.format(
            train_acc, epoch)
        opts.logger(_curr_str)
        _tmp += _curr_str.replace('\t', '&emsp;') + '<br/>'
        opts.logger('')
        if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
            info = {'msg': _tmp}
            vis.show_dynamic_info(phase='test', **info)

    _tmp = _tmp.replace('<b>TEST</b>', 'Last test stats')
    if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
        info = {'msg': _tmp}
        vis.show_dynamic_info(phase='test_finish', **info)

    # SAVE MODEL
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        last_epoch = epoch
        last_iter = step
        model_weights = net.module.state_dict() if opts.ctrl.multi_gpu else net.state_dict()
        file_to_save = {
            'state_dict': model_weights,
            'lr': new_lr,
            'epoch': epoch,
            'iter': step,
            'val_acc': accuracy,
            'options': opts,
        }
        if opts.misc.vis.use and opts.misc.vis.method == 'visdom':
            file_to_save.update({'loss_data': vis.loss_data})
            info = {
                'acc': accuracy,
                'epoch': epoch,
                'iter': step,
                'path': opts.io.model_file
            }
            vis.show_best_model(**info)

        torch.save(file_to_save, opts.io.model_file)
        opts.logger('\tBest model saved to: {}, at [epoch {} / iter {}]\n'.format(
            opts.io.model_file, epoch, step))

        return [best_accuracy, last_epoch, last_iter]
    else:
        return [-1]
    # DONE WITH SAVE MODEL
