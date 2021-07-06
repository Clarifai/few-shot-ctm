import torch
from copy import deepcopy


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

    net.eval()
    with torch.no_grad():
        for j, batch_test in enumerate(input_db):

            if j >= eval_length:
                break

            support_x, support_y, query_x, query_y = process_input(batch_test, opts, mode='test')

            if opts.fsl.ctm:
                _, correct = net.forward_CTM(support_x, support_y, query_x, query_y, False)
            else:
                if opts.model.structure == 'original':
                    support_x, support_y, query_x, query_y = \
                        support_x.squeeze(0), support_y.squeeze(0), query_x.squeeze(0), query_y.squeeze(0)
                    _, correct = net(support_x, support_y, query_x, query_y, False)
                else:
                    _, correct = net(support_x, support_y, query_x, query_y, False,
                                     opts.fsl.n_way[which_ind], curr_shot)

            # multi-gpu support
            total_correct += correct.sum().float()
            total_num += query_y.numel()

        # due to python 2, it's converted to int!
        accuracy = total_correct / total_num
        accuracy = accuracy.item()
    net.train()
    return accuracy


# def test_model_pretrain(net, input_db, eval_length, opts):
#
#     total_correct, total_num, display_onebatch = \
#         torch.zeros(1).to('cuda'), torch.zeros(1).to('cuda'), False
#
#     net.eval()
#     with torch.no_grad():
#         for j, batch_test in enumerate(input_db):
#
#             if j >= eval_length:
#                 break
#
#             x, y = batch_test[0].to(opts.ctrl.device), batch_test[1].to(opts.ctrl.device)
#             predict = net(x).argmax(dim=1, keepdim=True)
#             correct = torch.eq(predict, y)
#
#             # compute correct
#             total_correct += correct.sum().float()  # multi-gpu support
#             total_num += predict.numel()
#
#     accuracy = total_correct / total_num  # due to python 2, it's converted to int!
#     accuracy = accuracy.item()
#     net.train()
#     return accuracy


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

    _curr_str = '\tEvaluating at epoch {}, step {}, with eval_length {} ... (be patient)'.format(
        epoch, step, int(eval_length))
    opts.logger(_curr_str)

    accuracy = test_model(net, val_db, eval_length, opts, which_ind, curr_shot, optimizer, meta_test)

    eqn = '>' if accuracy > best_accuracy else '<'
    _curr_str = '\t\tCurrent accuracy is {:.4f} {} ' \
                'previous best accuracy is {:.4f} (ep{}, iter{})'.format(
        accuracy, eqn, best_accuracy, last_epoch, last_iter)
    opts.logger(_curr_str)

    # Also test the train-accuracy at end of one epoch
    if opts.test.compute_train_acc and step == total_iter - 1 and not opts.ctrl.eager:
        _curr_str = '\tEvaluating training acc at epoch {}, step {}, length {} ... (be patient)'.format(
            epoch, step, len(train_db))
        opts.logger(_curr_str)

        train_acc = test_model(net, train_db, len(train_db), opts, which_ind, curr_shot, optimizer, meta_test)

        _curr_str = '\t\tCurrent train_accuracy is {:.4f} at END of epoch {:d}'.format(
            train_acc, epoch)
        opts.logger(_curr_str)
        opts.logger('')

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

        torch.save(file_to_save, opts.io.model_file)
        opts.logger('\tBest model saved to: {}, at [epoch {} / iter {}]\n'.format(
            opts.io.model_file, epoch, step))

        return [best_accuracy, last_epoch, last_iter]
    else:
        return [-1]
    # DONE WITH SAVE MODEL

