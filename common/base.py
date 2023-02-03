import abc
import glob
import math
import os.path as osp

import torch.optim
import torchvision.transforms as transforms
from config import cfg
from dataset import MultipleDatasets
from logger import colorlogger
from timer import Timer
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader

from data.DexYCB import DexYCB
from data.HO3D import HO3D


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self, args):
        return

    @abc.abstractmethod
    def _make_model(self, args):
        return

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        # total_params = []
        # print(model.module)
        # assert False
        # for module in model.module.trainable_modules:
        #     total_params += list(module.parameters())
        # optimizer = torch.optim.Adam(total_params, lr=cfg.lr)
        # return optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        print("loading pretrained model")
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        # cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        cur_epoch = 10
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        #optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            dataset_class = eval(f"{cfg.trainset_3d[i]}.{cfg.trainset_3d[i]}")
            if 'HO3D' in cfg.trainset_3d[i]:
                dataset = dataset_class(split='train')
                trainset3d_loader.append(dataset)
                print(f"Included {len(dataset)} samples")
            if 'DexYCB' in cfg.trainset_3d[i]:
                dataset = dataset_class(split='train')
                trainset3d_loader.append(dataset)
                print(f"Included {len(dataset)} samples")
                # from data.DexYCB.DexYCB import filter_index
                # for i in tqdm(range(len(dataset))):
                #     dataset[i]
                # with open('../data/DexYCB/Dex_train_filter.pth', 'wb') as f:
                #     pickle.dump(filter_index, f)
                # assert False
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))
        
        if len(trainset3d_loader) > 0 and len(trainset2d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset3d_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
            trainset2d_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
            trainset_loader = MultipleDatasets([trainset3d_loader, trainset2d_loader], make_same_len=True)
        elif len(trainset3d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
        elif len(trainset2d_loader) > 0:
            self.vertex_num = trainset2d_loader[0].vertex_num
            self.joint_num = trainset2d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
        else:
            assert 0, "Both 3D training set and 2D training set have zero length."
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)

    def _make_batch_generator_distributed(self, args):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        dataset_class = eval(f"{cfg.trainset[i]}.{cfg.trainset[i]}")
        train_loader = dataset_class(split='train')

        sampler = torch.utils.data.distributed.DistributedSampler(
            train_loader,
            num_replicas=cfg.num_gpus,
            rank=args.local_rank,
            shuffle=False
        )

        batch_generator = DataLoader(dataset=train_loader, batch_size=cfg.num_gpus,
                                     sampler=sampler, shuffle=False,
                                     num_workers=cfg.num_thread, pin_memory=True, drop_last=False)

        self.testset = train_loader
        self.vertex_num = train_loader.vertex_num
        self.joint_num = train_loader.joint_num
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        from model_no_render import get_model

        model = get_model(self.vertex_num, self.joint_num, 'train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if True:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

class Tester(Base):
    def __init__(self):
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset3d_loader = []
        if 'HO3D' == cfg.testset:
            testset3d_loader.append(HO3D.HO3D(split='test'))
            print(f"Included {len(testset3d_loader[0])} samples")

        if 'DexYCB' == cfg.testset:
            dataset = DexYCB.DexYCB(split='test')
            testset3d_loader.append(dataset)
            print(f"Included {len(dataset)} samples")

            # from data.DexYCB.DexYCB import filter_index
            # print("filtering ")
            # for i in tqdm(range(len(dataset))):
            #     dataset[i]
            # with open('../data/DexYCB/Dex_test_filter_1cm.pth', 'wb') as f:
            #         pickle.dump(filter_index, f)
            # print(f"filtering {len(filter_index)}, remaining {len(dataset) - len(filter_index)}")
            # assert False
        
        if len(testset3d_loader) > 0:
            self.vertex_num = testset3d_loader[0].vertex_num
            self.joint_num = testset3d_loader[0].joint_num
            testset3d_loader = MultipleDatasets(testset3d_loader, make_same_len=False)
        else:
            assert 0, "Failed to load test set."
        self.itr_per_epoch = math.ceil(len(testset3d_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=testset3d_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

    def _make_model(self, path):
        model_path = cfg.saved_model_path if path is None else path
        # model_dir = cfg.model_dir
        # last_pth = sorted(os.listdir(model_dir))[-1]
        # model_path = os.path.join(model_dir, last_pth)
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        from model_no_render import get_model

        model = get_model(self.vertex_num, self.joint_num, 'test')

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DataParallel(model).cuda()

        ckpt = torch.load(model_path)

        model.load_state_dict(ckpt['network'], strict=False) # some layers are cleaned in the final version

        # print("param for obj_mesh_net", torch.norm(torch.nn.utils.parameters_to_vector(model.module.obj_mesh_net.parameters())))
        # print("param for hand_mesh_net", torch.norm(torch.nn.utils.parameters_to_vector(model.module.mesh_net.parameters())))
        
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)
