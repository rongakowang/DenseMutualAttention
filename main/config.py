import os
import os.path as osp
import sys

sys.path.append('..')
from common.utils.dir import add_pypath, make_folder


class Config:
    
    ## dataset
    trainset_3d = [] # HO3D, DexYCB
    trainset_2d = []
    testset = 'DexYCB' # HO3D, DexYCB

    ## model setting
    resnet_type = 50
    
    ## input, output
    input_img_shape = (256, 256)
    output_hm_shape = (64, 64, 64)
    bbox_3d_size = 0.3
    sigma = 2.5

    ## training config
    lr_dec_epoch = [10, 20]
    end_epoch = 25
    lr = 1e-4
    lr_dec_factor = 10
    train_batch_size = 24
    sil_loss_weight = 10
    loss_type = 'l1'

    ycb_ho3d_root = "../local_data/ho3d_simple"
    ycb_dex_root = '../local_data/dex_simple'
    dataset_root = "../local_data/"
    mano_root = "../local_data/mano"
    
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    
    # version control
    simple_meshnet = True # replace the meshnet with simple_meshnet to reduce computation
    bbox_crop = True # crop hand and object

    ## testing config
    test_batch_size = 1
    use_gt_info = False

    ## others
    num_thread = 4
    gpu_ids = '0,1'
    num_gpus = 2
    stage = 'param' # lixel, param
    continue_train = False
    rnn = False
    
    ## directory
    data_dir = osp.join(root_dir, '../dataset/')
    output_dir = osp.join(root_dir, 'output')
    
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    smpl_path = osp.join(root_dir, 'common', 'utils', 'smplpytorch')
    
    def set_args(self, gpu_ids, stage='lixel', test_set='HO3D', continue_train=False, rnn=False, tex=False, finetune=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.stage = stage
        # extend training schedule
        if self.stage == 'param':
            self.lr_dec_epoch = [x+5 for x in self.lr_dec_epoch]
            self.end_epoch = self.end_epoch + 5
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

        self.rnn = rnn
        self.tex = tex
        self.finetune = finetune
        self.testset = test_set


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
add_pypath(osp.join(cfg.root_dir))
add_pypath(osp.join(cfg.data_dir))
add_pypath('../..')
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
