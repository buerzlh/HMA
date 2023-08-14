import torch
import torch.nn as nn
import os
from . import utils as solver_utils 
from utils.utils import to_cuda, mean_accuracy, accuracy
from torch import optim
from math import ceil as ceil
from config.config import cfg

class BaseSolver:
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        self.opt = cfg
        self.source_name = self.opt.DATASET.SOURCE_NAME
        self.target_name = self.opt.DATASET.TARGET_NAME

        self.net = net
        net = net.module
        if(len(net.FC)!=0):
            self.F = nn.Sequential(
                net.feature_extractor,
                net.FC
                )
        else:
            self.F = net.feature_extractor
        self.C = net.classifier
        self.init_data(dataloader)

        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda() 

        self.loop = 0
        self.iters = 0
        self.iters_per_loop = None
        self.history = {}

        self.base_lr = self.opt.TRAIN.BASE_LR
        self.momentum = self.opt.TRAIN.MOMENTUM

        self.bn_domain_map = bn_domain_map

        self.optim_state_dict = None
        self.resume = False
        if resume is not None:
            self.resume = True
            self.loop = resume['loop']
            self.iters = resume['iters']
            self.history = resume['history']
            self.optim_state_dict = resume['optimizer_state_dict']
            self.bn_domain_map = resume['bn_domain_map']
            print('Resume Training from loop %d, iters %d.' % \
			(self.loop, self.iters))

        self.build_optimizer()

    def init_data(self, dataloader):
        self.train_data = {key: dict() for key in dataloader if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloader:
                continue
            cur_dataloader = dataloader[key]
            self.train_data[key]['loader'] = cur_dataloader 
            self.train_data[key]['iterator'] = None

        if 'test' in dataloader:
            self.test_data = dict()
            self.test_data['loader'] = dataloader['test']
        
        
    def build_optimizer(self):
        self.optimizer={}
        self.lr_scheduler= {}
        opt = self.opt
        if(len(self.net.module.FC)!=0):
            param_groups_f = [
                {'params':self.net.module.feature_extractor.parameters(),'lr_mult': 1.0},
                {'params':self.net.module.FC.parameters(),'lr_mult': opt.TRAIN.LR_MULT}           
                            ]
        else:
            param_groups_f = [
                {'params':self.net.module.feature_extractor.parameters(),'lr_mult': 1.0}         
                            ]
        param_groups_c = [
                {'params':self.net.module.classifier.parameters(),'lr_mult': opt.TRAIN.LR_MULT}         
                            ]
        

        # param_groups = solver_utils.set_param_groups(self.net, 
		# dict({'FC': opt.TRAIN.LR_MULT}))

        assert opt.TRAIN.OPTIMIZER in ["Adam", "SGD"], \
            "Currently do not support your specified optimizer."

        if opt.TRAIN.OPTIMIZER == "Adam":
            self.optimizer['F'] = optim.Adam(param_groups_f, 
			lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2], 
			weight_decay=opt.TRAIN.WEIGHT_DECAY)
            self.optimizer['C'] = optim.Adam(param_groups_c, 
			lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2], 
			weight_decay=opt.TRAIN.WEIGHT_DECAY)
            self.lr_scheduler['F'] = self.build_lr_scheduler(self.optimizer['F'],opt.TRAIN)
            self.lr_scheduler['C'] = self.build_lr_scheduler(self.optimizer['C'],opt.TRAIN)
        elif opt.TRAIN.OPTIMIZER == "SGD":
            self.optimizer['F'] = optim.SGD(param_groups_f, 
			lr=self.base_lr, momentum=self.momentum, 
			weight_decay=opt.TRAIN.WEIGHT_DECAY)
            self.optimizer['C'] = optim.SGD(param_groups_c, 
			lr=self.base_lr, momentum=self.momentum, 
			weight_decay=opt.TRAIN.WEIGHT_DECAY)
            self.lr_scheduler['F'] = self.build_lr_scheduler(self.optimizer['F'],opt.TRAIN)
            self.lr_scheduler['C'] = self.build_lr_scheduler(self.optimizer['C'],opt.TRAIN)        

    def build_lr_scheduler(self, optimizer, optim_cfg):
        """A function wrapper for building a learning rate scheduler.

        Args:
            optimizer (Optimizer): an Optimizer.
            optim_cfg (CfgNode): optimization config.
        """
        AVAI_SCHEDS = ['single_step', 'multi_step', 'cosine']
        lr_scheduler = optim_cfg.LR_SCHEDULER
        stepsize = optim_cfg.STEPSIZE
        gamma = optim_cfg.GAMMA
        max_epoch = optim_cfg.MAX_LOOP

        if lr_scheduler not in AVAI_SCHEDS:
            raise ValueError(
                'Unsupported scheduler: {}. Must be one of {}'.format(
                    lr_scheduler, AVAI_SCHEDS
                )
            )

        if lr_scheduler == 'single_step':
            if isinstance(stepsize, (list, tuple)):
                stepsize = stepsize[-1]

            if not isinstance(stepsize, int):
                raise TypeError(
                    'For single_step lr_scheduler, stepsize must '
                    'be an integer, but got {}'.format(type(stepsize))
                )

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=stepsize, gamma=gamma
            )

        elif lr_scheduler == 'multi_step':
            if not isinstance(stepsize, list):
                raise TypeError(
                    'For multi_step lr_scheduler, stepsize must '
                    'be a list, but got {}'.format(type(stepsize))
                )

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=stepsize, gamma=gamma
            )

        elif lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(max_epoch)
            )

        return scheduler

    def update_lr(self):
        # for key in self.lr_scheduler.keys():
        #     self.lr_scheduler[key].step()
        iters = self.iters
        if self.opt.TRAIN.LR_SCHEDULE == 'exp':
            solver_utils.adjust_learning_rate_exp(self.base_lr, 
			self.optimizer['F'], iters, 
                        decay_rate=self.opt.EXP.LR_DECAY_RATE,
			decay_step=self.opt.EXP.LR_DECAY_STEP)
            solver_utils.adjust_learning_rate_exp(self.base_lr, 
			self.optimizer['C'], iters, 
                        decay_rate=self.opt.EXP.LR_DECAY_RATE,
			decay_step=self.opt.EXP.LR_DECAY_STEP)

        elif self.opt.TRAIN.LR_SCHEDULE == 'inv':
            solver_utils.adjust_learning_rate_inv(self.base_lr, self.optimizer['F'], 
		    iters, self.opt.INV.ALPHA, self.opt.INV.BETA)
            solver_utils.adjust_learning_rate_inv(self.base_lr, self.optimizer['C'], 
		    iters, self.opt.INV.ALPHA, self.opt.INV.BETA)

        else:
            raise NotImplementedError("Currently don't support the specified \
                    learning rate schedule: %s." % self.opt.TRAIN.LR_SCHEDULE)

    def logging(self, loss, accu):
        print('[loop: %d, iters: %d]: ' % (self.loop, self.iters))
        loss_names = ""
        loss_values = ""
        for key in loss:
            loss_names += key + ","
            loss_values += '%.4f,' % (loss[key])
        loss_names = loss_names[:-1] + ': '
        loss_values = loss_values[:-1] + ';'
        loss_str = loss_names + loss_values + (' source %s: %.4f.' % 
                    (self.opt.EVAL_METRIC, accu))
        print(loss_str)

    def model_eval(self, preds, gts):
        assert(self.opt.EVAL_METRIC in ['mean_accu', 'accuracy']), \
             "Currently don't support the evaluation metric you specified."

        if self.opt.EVAL_METRIC == "mean_accu": 
            res = mean_accuracy(preds, gts)
        elif self.opt.EVAL_METRIC == "accuracy":
            res = accuracy(preds, gts)
        return res

    def save_ckpt(self):
        save_path = self.opt.SAVE_DIR
        # ckpt_weights = os.path.join(save_path, 'ckpt_%d_%d.weights' % (self.loop, self.iters))
        ckpt_weights = os.path.join(save_path, 'ckpt_max.weights')

        torch.save({'weights': self.net.module.state_dict(),
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_weights)


    def complete_training(self):
        if self.loop > self.opt.TRAIN.MAX_LOOP:
            return True

    def register_history(self, key, value, history_len):
        if key not in self.history:
            self.history[key] = [value]
        else:
            self.history[key] += [value]
        
        if len(self.history[key]) > history_len:
            self.history[key] = \
                 self.history[key][len(self.history[key]) - history_len:]
       
    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name 

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample

    def get_samples_categorical(self, data_name, category):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader'][category]
        data_iterator = self.train_data[data_name]['iterator'][category]
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'][category] = data_iterator

        return sample

    def test(self):
        self.net.eval()
        preds = []
        gts = []
        for sample in iter(self.test_data['loader']):
            data, gt = to_cuda(sample['Img']), to_cuda(sample['Label'])
            logits = self.net(data)['logits']
            preds += [logits]
            gts += [gt]

        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)

        res = self.model_eval(preds, gts)
        return res

    def clear_history(self, key):
        if key in self.history:
            self.history[key].clear()

    def solve(self):
        pass

    def update_network(self, **kwargs):
        pass