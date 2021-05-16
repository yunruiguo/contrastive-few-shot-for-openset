import time
import os.path as osp
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm


def calc_auroc(known_scores, unknown_scores):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)

    return auc_score

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model, self.trlog = prepare_model(args, self.trlog)
        self.optimizer, self.lr_scheduler, self.optimizer_margin, self.lr_scheduler_margin = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux

    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()

            for batch in tqdm(self.train_loader):

                data, gt_label = [_.cuda() for _ in batch]
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                logits, reg_logits = self.para_model(data)
                logits = logits.view(-1, args.way)
                oh_query = torch.nn.functional.one_hot(label, args.way)

                sims = logits
                temp = (sims * oh_query).sum(-1)
                e_sim_p = temp - self.model.margin
                e_sim_p_pos = F.relu(e_sim_p)
                e_sim_p_neg = F.relu(-e_sim_p)

                l_open_margin = args.open_balance * e_sim_p_pos.mean(-1)
                l_open = args.open_balance * e_sim_p_neg.mean(-1)
                if reg_logits is not None:
                    loss = F.cross_entropy(logits, label)
                    reg_loss = args.balance * F.cross_entropy(reg_logits, label_aux)
                    total_loss = loss + reg_loss
                else:
                    loss = F.cross_entropy(logits, label)
                total_loss = total_loss + l_open
                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward(torch.randn_like(total_loss), retain_graph=True)
                torch.cuda.synchronize()
                self.optimizer_margin.zero_grad()

                l_open_margin.backward(torch.randn_like(l_open_margin))

                self.optimizer.step()
                self.optimizer_margin.step()

                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)
                # refresh start_tm
                start_tm = time.time()

            print('lr: {:.4f} Total_loss: {:.4f} ce_loss {:.4f} l_open: {:4f} R: {:4f} aux_loss: {:4f}'.format(self.optimizer_margin.param_groups[0]['lr'],\
                total_loss.item(), loss.item(), l_open.item(), self.model.margin.item(), reg_loss))
            self.lr_scheduler.step()
            self.lr_scheduler_margin.step()

            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def open_evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((len(data_loader), 4))  # loss and acc

        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('Evaluating ... best epoch {}, SnaTCHer={:.4f} + {:.4f}  acc={:.4f} + {:.4f}'.format(
            self.trlog['max_auc_epoch'],
            self.trlog['max_auc'],
            self.trlog['max_auc_interval'],
            self.trlog['acc'],
            self.trlog['acc_interval']))

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):

                data, _ = [_.cuda() for _ in batch]

                logits = self.para_model(data)
                logits = logits.reshape([-1, args.eval_way + args.open_eval_way, args.way])
                klogits = logits[:, :args.eval_way, :].reshape(-1, args.way)
                ulogits = logits[:, args.eval_way:, :].reshape(-1, args.way)
                loss = F.cross_entropy(klogits, label)
                acc = count_acc(klogits, label)

                """ Distance """
                kdist = -(klogits.max(1)[0])
                udist = -(ulogits.max(1)[0])
                kdist = kdist.cpu().detach().numpy()
                udist = udist.cpu().detach().numpy()
                dist_auroc = calc_auroc(kdist, udist)

                """ Snatcher """
                with torch.no_grad():
                    instance_embs = self.para_model.instance_embs
                    support_idx = self.para_model.support_idx
                    query_idx = self.para_model.query_idx

                    support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
                    query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
                    emb_dim = support.shape[-1]

                    support = support[:, :, :args.way].contiguous()
                    # get mean of the support
                    bproto = support.mean(dim=1)  # Ntask x NK x d
                    proto = self.para_model.slf_attn(bproto, bproto, bproto)
                    kquery = query[:, :, :args.way].contiguous()
                    uquery = query[:, :, args.way:].contiguous()
                    snatch_known = []
                    for j in range(75):
                        pproto = bproto.clone().detach()
                        """ Algorithm 1 Line 1 """
                        c = klogits.argmax(1)[j]
                        """ Algorithm 1 Line 2 """
                        pproto[0][c] = kquery.reshape(-1, emb_dim)[j]
                        """ Algorithm 1 Line 3 """
                        pproto = self.para_model.slf_attn(pproto, pproto, pproto)[0]
                        pdiff = (pproto - proto).pow(2).sum(-1).sum() / 64.0
                        """ pdiff: d_SnaTCHer in Algorithm 1 """
                        snatch_known.append(pdiff)

                    snatch_unknown = []
                    for j in range(ulogits.shape[0]):
                        pproto = bproto.clone().detach()
                        """ Algorithm 1 Line 1 """
                        c = ulogits.argmax(1)[j]
                        """ Algorithm 1 Line 2 """
                        pproto[0][c] = uquery.reshape(-1, emb_dim)[j]
                        """ Algorithm 1 Line 3 """
                        pproto = self.para_model.slf_attn(pproto, pproto, pproto)[0]
                        pdiff = (pproto - proto).pow(2).sum(-1).sum() / 64.0
                        """ pdiff: d_SnaTCHer in Algorithm 1 """
                        snatch_unknown.append(pdiff)

                    pkdiff = torch.stack(snatch_known)
                    pudiff = torch.stack(snatch_unknown)
                    pkdiff = pkdiff.cpu().detach().numpy()
                    pudiff = pudiff.cpu().detach().numpy()

                    snatch_auroc = calc_auroc(pkdiff, pudiff)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc
                record[i - 1, 2] = snatch_auroc
                record[i - 1, 3] = dist_auroc

        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        auc_sna, auc_sna_p = compute_confidence_interval(record[:, 2])
        auc_dist, auc_dist_p = compute_confidence_interval(record[:, 3])
        print("acc: {:.4f} + {:.4f} Dist: {:.4f} + {:.4f} SnaTCHer: {:.4f} + {:.4f}" \
              .format(va, vap, auc_dist, auc_dist_p, auc_sna, auc_sna_p))
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap, auc_sna, auc_sna_p
    def evaluate_test(self):

        # evaluation mode
        # self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_auc.pth'))['params'])
        # self.model.load_state_dict(torch.load(osp.join(self.args.init_weights))['params'])
        self.model.eval()

        vl, va, vap, auc_sna, auc_sna_p = self.open_evaluate(self.test_loader)
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_auc'] = auc_sna
        self.trlog['test_auc_interval'] = auc_sna_p


        print('Test acc={:.4f} + {:.4f}   Test auc={:.4f} + {:.4f}\n'.format(
            self.trlog['test_acc'],
            self.trlog['test_acc_interval'],
            self.trlog['test_auc'],
            self.trlog['test_auc_interval']))

        return vl, va, vap
    
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('Best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_auc_epoch'],
                self.trlog['max_auc'],
                self.trlog['max_auc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))