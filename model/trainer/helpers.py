import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler, RandomSampler, ClassSampler
from model.models.protonet import ProtoNet
from model.models.matchnet import MatchNet
from model.models.feat import FEAT
from model.models.featstar import FEATSTAR
from model.models.deepset import DeepSet
from model.models.bilstm import BILSTM
from model.models.graphnet import GCN
from model.models.semi_feat import SemiFEAT
from model.models.semi_protofeat import SemiProtoFEAT

class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])
                
                yield ( torch.cat(_, dim=0) for _ in output_batch )
            except StopIteration:
                done = True
        return

def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch*num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers=args.num_workers*num_device if args.multi_gpu else args.num_workers
    trainset = Dataset('train', args, augment=args.augment)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                      num_episodes,
                                      max(args.way, args.num_classes),
                                      args.shot + args.query)

    train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                            args.num_eval_episodes,
                            args.eval_way + args.open_eval_way, args.eval_shot + args.eval_query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)

    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                            args.num_test_episodes,
                            args.eval_way + args.open_eval_way, args.eval_shot + args.eval_query)
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)    

    return train_loader, val_loader, test_loader

def prepare_model(args, trlog):
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    trlog['acc'] = 0
    trlog['acc_interval'] = 0
    if args.init_weights > -1:

        model_dict = model.state_dict()
        if args.init_weights == 0:
            pretrained_dict = torch.load(args.save_path + '/max_auc.pth')['params']
            trlog = torch.load(os.path.join(args.save_path, 'trlog'))
        elif args.init_weights == 1:
            pretrained_dict = torch.load('./saves/' + args.dataset + '/feat-{}-shot.pth'.format(args.shot))['params']
        if args.backbone_class == 'ConvNet':
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print('Current device: ', torch.cuda.current_device())
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cuda()
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, device_ids=[0, 1])
        # para_model = nn.DataParallel(model, device_ids=[0, 1])
        para_model = model.cuda()
    else:
        para_model = model.cuda()

    return model, para_model, trlog

def prepare_optimizer(model, args):
    top_para = [v for k,v in model.named_parameters() if 'encoder' not in k and 'margin' not in k and 'recipro' not in k]
    margin_para = [v for k,v in model.named_parameters() if 'margin' in k or 'recipro' in k]

    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone_class in ['ConvNet']:
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )                
    else:
        optimizer = optim.SGD(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )
        optimizer_margin = optim.SGD(
            [{'params': margin_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
        lr_scheduler_margin = optim.lr_scheduler.StepLR(
            optimizer_margin,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
        lr_scheduler_margin = optim.lr_scheduler.MultiStepLR(
            optimizer_margin,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler, optimizer_margin, lr_scheduler_margin
