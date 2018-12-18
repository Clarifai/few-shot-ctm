import numpy as np
from tools.general_utils import Logger, AttrDict


class Visualizer(object):
    def __init__(self, opt, loss_data=None):
        self.opt = opt
        import visdom

        prefix = 'eager_' if opt.ctrl.eager else ''
        env_name = prefix + opt.ctrl.yaml_file.replace('/', '_') + '_eid_' + str(opt.ctrl.eid)
        self.visualizer = visdom.Visdom(port=opt.misc.vis.port, env=env_name)
        self.dis_im_cnt, self.dis_im_cycle = 0, 4

        if loss_data is not None:
            self.loss_data = loss_data
            assert len(self.loss_data['legend']) == len(self.opt.misc.vis.loss_legend)
        else:
            self.loss_data = {'X': [], 'Y': [], 'legend': self.opt.misc.vis.loss_legend}
        self.line = dict()  # for loss
        self.line['height'] = 500
        self.line['width'] = 800

        self.txt = dict()  # for _show_config
        self.txt['height'] = 500
        self.txt['width'] = 320

        self.start_epoch = opt.ctrl.start_epoch
        self.start_iter = opt.ctrl.start_iter
        # self.mAP_msg = 'Config name:' \
        #                '<br/>&emsp;{:s}<br/><br/>'.format(self.opt.ctrl.yaml_file)
        self.train_msg = ''
        self.test_msg = ''
        self._show_config()

    def _show_config(self):
        """show config on visdom console"""

        def _print_attr_dict(k, v, indent):
            _msg = '{:s}<u>{:s}</u>:<br/>'.format(indent, k)
            for _k, _v in sorted(v.items()):
                if isinstance(_v, AttrDict):
                    _msg += _print_attr_dict(_k, _v, indent=indent + '&emsp;')
                elif isinstance(_v, Logger):
                    _msg += '{:s}&emsp;{:s}:&emsp;&emsp;&emsp;{}<br/>'.format(indent, _k, 'Class object not shown here')
                else:
                    _msg += '{:s}&emsp;{:s}:&emsp;&emsp;&emsp;<b>{}</b><br/>'.format(indent, _k, _v)
            return _msg

        msg = ''
        temp = self.opt.__dict__
        for k in sorted(temp):
            if isinstance(temp[k], AttrDict):
                msg += _print_attr_dict(k, temp[k], indent='')
            elif isinstance(temp[k], bool):
                msg += '<u>{}</u>:&emsp;&emsp;<b>{}</b><br/>'.format(k, temp[k])
            else:
                msg += '<u>{}</u>:&emsp;&emsp;{}<br/>'.format(k, 'Class object not shown here')

        self.visualizer.text(
            msg,
            opts={
                'title': 'Configurations',
                'height': self.txt['height'],
                'width': self.txt['width']},
            win=self.opt.misc.vis.txt)

    def plot_loss(self, **args):
        """draw loss on visdom console"""
        curr_ep, curr_iter, total_iter = args['curr_ep'], args['curr_iter'], args['total_iter']
        y_num = len(self.loss_data['legend'])

        loss = args['loss']
        if loss.ndim == 0:
            loss = [loss]

        if self.opt.fsl.swap:
            x_progress = [curr_ep + curr_iter * 1.0 / total_iter for _ in range(y_num/2)]
            _extend = [self.opt.fsl.swap_num * x_progress[0] for _ in range(y_num/2)]
            x_progress.extend(_extend)
        else:
            x_progress = [curr_ep + curr_iter * 1.0 / total_iter for _ in range(y_num)]

        if self.opt.fsl.swap:
            loss.extend(loss)

        # loss_list = [loss[i] for i in range(y_num)]
        loss_list = loss
        self.loss_data['X'].append(x_progress)
        self.loss_data['Y'].append(loss_list)
        self.visualizer.line(
            X=np.array(self.loss_data['X']),
            Y=np.array(self.loss_data['Y']),
            opts={
                'title': 'Train loss over epoch',
                'legend': self.loss_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss',
                'height': self.line['height'],
                'width': self.line['width']
            },
            win=self.opt.misc.vis.line,
        )

    def show_dynamic_info(self, phase='train', **args):
        """show dynamic info on visdom console"""
        if phase == 'train':
            curr_ep, iter_ind, total_iter, total_epoch = \
                args['curr_ep'], args['curr_iter'], args['total_iter'], args['total_ep']
            days, hrs, total_days, iter_time = \
                args['left_time'][0], args['left_time'][1], args['left_time'][2], args['iter_time']

            if days == 0 and hrs < 2:
                status = 'SOON'
            else:
                status = 'RUNNING'

            msg = 'Status: <b>{:s}</b><br/>'.format(status)
            dynamic = 'Start epoch: {:d}, iter: {:d}<br/>' \
                      'Current lr: {:.8f}<br/>' \
                      'Progress: <br/>&emsp;<b>epoch {:d} / {:d}, iter {:06d}/{}</b><br/><br/>' \
                      'est. left time: {:d} days, {:.2f} hrs, &emsp;total time taken: {:.3f} days<br/>' \
                      'time per "image" (iter_time/bs): {:.4f} sec<br/>'.format(
                        self.start_epoch, self.start_iter,
                        args['lr'],
                        curr_ep, total_epoch, iter_ind, total_iter,
                        days, hrs, total_days,
                        iter_time / self.opt.train.batch_sz)
            self.train_msg = msg + dynamic
            msg_to_show = self.train_msg + self.test_msg
        # elif args['type'] == 'Runtime Error':
        #     error_str = '<br/><br/><b>ERROR OCCURS at epoch {:d}, iter {:d} !!!</b>'\
        #         .format(args['curr_ep'], args['iter_ind'])
        #     curr_msg = self.train_msg + error_str
        # elif args['type'] == 'Keyboard Interrupt':
        #     error_str = '<br/><br/><b>KEYBOARD INTERRUPT at epoch {:d} !!!</b>'\
        #         .format(args['curr_ep'])
        #     curr_msg = self.train_msg + error_str
        elif phase == 'train_finish':
            msg_to_show = self.train_msg.replace('SOON', 'DONE') + self.test_msg
        elif phase == 'test':
            msg_to_show = self.train_msg + self.test_msg + args['msg']
        elif phase == 'test_finish':
            self.test_msg = args['msg']
            msg_to_show = self.train_msg + self.test_msg
        elif phase == 'error':
            msg_to_show = self.train_msg + self.test_msg
            msg_to_show.replace('SOON', 'STOPPED')
            msg_to_show.replace('RUNNING', 'STOPPED')

        self.visualizer.text(
            msg_to_show,
            opts={
                'title': 'Train dynamics',
                'height': 500,
                'width': 300
            },
            win=self.opt.misc.vis.txt+1)

    def show_best_model(self, **args):
        curr = 'Best model:<br/>' \
               '&emsp;accuracy: <b>{:.4f}</b><br/>' \
               '&emsp;at epoch {}, iter {}<br/>' \
               '&emsp;model_path: {:s}'.format(
                    args['acc'], args['epoch'], args['iter'], args['path'])
        self.visualizer.text(
            curr,
            opts={
                'title': 'Test result',
                'height': 500,
                'width': 300
            },
            win=self.opt.misc.vis.txt+2)

    def save(self):
        self.visualizer.save([self.visualizer.env])

    # def show_image(self, progress, others=None):
    #     """for test, print log info in console and show detection results on visdom"""
    #     if self.opt.phase == 'test':
    #         name = os.path.basename(os.path.dirname(self.opt.det_file))
    #         i, total_im, test_time = progress[0], progress[1], progress[2]
    #         all_boxes, im, im_name = others[0], others[1], others[2]
    #
    #         print_log('[{:s}][{:s}]\tim_detect:\t{:d}/{:d} {:.3f}s'.format(
    #             self.opt.experiment_name, name, i, total_im, test_time), self.opt.file_name)
    #
    #         dets = np.asarray(all_boxes)
    #         result_im = self._show_detection_result(im, dets[:, i], im_name)
    #         result_im = np.moveaxis(result_im, 2, 0)
    #         win_id = self.dis_win_id_im + (self.dis_im_cnt % self.dis_im_cycle)
    #         self.vis.image(result_im, win=win_id,
    #                        opts={
    #                            'title': 'subfolder: {:s}, name: {:s}'.format(
    #                                os.path.basename(self.opt.save_folder), im_name),
    #                            'height': 320,
    #                            'width': 400,
    #                        })
    #         self.dis_im_cnt += 1
    #
    # def _show_detection_result(self, im, results, im_name):
    #
    #     plt.figure()
    #     plt.axis('off')     # TODO (irrelevant), still the axis remains
    #     plt.imshow(im)
    #     currentAxis = plt.gca()
    #
    #     for cls_ind in range(1, len(results)):
    #         if results[cls_ind] == []:
    #             continue
    #         else:
    #
    #             cls_name = self.class_name[cls_ind-1]
    #             cls_color = self.color[cls_ind-1]
    #             inst_num = results[cls_ind].shape[0]
    #             for inst_ind in range(inst_num):
    #                 if results[cls_ind][inst_ind, -1] >= self.opt.visualize_thres:
    #
    #                     score = results[cls_ind][inst_ind, -1]
    #                     pt = results[cls_ind][inst_ind, 0:-1]
    #                     coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
    #                     display_txt = '{:s}: {:.2f}'.format(cls_name, score)
    #
    #                     currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=cls_color, linewidth=2))
    #                     currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': cls_color, 'alpha': .5})
    #                 else:
    #                     break
    #     result_file = '{:s}/{:s}.png'.format(self.save_det_res_path, im_name[:-4])
    #
    #     plt.savefig(result_file, dpi=300, bbox_inches="tight", pad_inches=0)
    #     plt.close()
    #     # ref: https://github.com/facebookresearch/visdom/issues/119
    #     # plotly_fig = tls.mpl_to_plotly(fig)
    #     # self.vis._send({
    #     #     data=plotly_fig.data,
    #     #     layout=plotly_fig.layout,
    #     # })
    #     result_im = imread(result_file)
    #     return result_im
