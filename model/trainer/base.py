import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from model.logger import Logger

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_auc'] = 0.0
        self.trlog['max_auc_epoch'] = 0
        self.trlog['max_auc_interval'] = 0.0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def open_evaluate(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass    
    
    @abc.abstractmethod
    def final_record(self):
        pass    

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            vl, va, vap, auc_sna, auc_sna_p = self.open_evaluate(self.val_loader)
            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(va),  self.train_epoch)
            print('epoch {}, val, auc={:.4f} acc={:.4f}+{:.4f}'.format(epoch, auc_sna, va, vap))

            if auc_sna >= self.trlog['max_auc']:
                self.trlog['max_auc'] = auc_sna
                self.trlog['max_auc_interval'] = auc_sna_p
                self.trlog['max_auc_epoch'] = self.train_epoch
                self.trlog['acc'] = va
                self.trlog['acc_interval'] = vap
                self.trlog['optim_lr'] = self.optimizer.param_groups[0]['lr']
                self.trlog['margin_optim_lr'] = self.optimizer_margin.param_groups[0]['lr']
                torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
                self.save_model('max_auc')


    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc',  ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('grad_norm',  tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, '     \
                  'forward_timer: {:.2f} sec,'   \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                        self.dt.item(), self.ft.item(),
                        self.bt.item(), self.ot.item())
                  )
            self.logger.dump()

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )


    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
