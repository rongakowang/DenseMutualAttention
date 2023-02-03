import argparse
import time
import warnings
from test import get_inter_metrics

import torch
import torch.backends.cudnn as cudnn
from config import cfg
from pytorch3d.structures import Meshes
from tqdm import tqdm

from common.base import Tester

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--model_path', type=str, dest='model_path')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

def get_boundingbox_corners(verts, face):
    mm = Meshes(verts=[verts], faces=[face], textures=None).cuda().get_bounding_boxes()[0]
    bbox = torch.Tensor([[mm[0,0], mm[1,0], mm[2,0]],
                         [mm[0,1], mm[1,0], mm[2,0]],
                         [mm[0,0], mm[1,1], mm[2,0]],
                         [mm[0,0], mm[1,0], mm[2,1]],
                         [mm[0,1], mm[1,1], mm[2,0]],
                         [mm[0,0], mm[1,1], mm[2,1]],
                         [mm[0,1], mm[1,0], mm[2,1]],
                         [mm[0,1], mm[1,1], mm[2,1]],]).cuda()
    return bbox


def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids, test_set="DexYCB")
    cudnn.benchmark = True

    tester = Tester()
    tester._make_batch_generator()
    tester._make_model(args.model_path)

    hand_mje = []
    verts_dist = []
    mean_penetration_depth = []
    avg_fps = []

    save_records = True

    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        with torch.no_grad():            
            start = time.time()
            out = tester.model(inputs, targets, meta_info, 'test')
            end = time.time()
            avg_fps.append((1/(end - start))) # calculate FPS

            hand_verts_out = out['hand_verts_out'][0]
            hand_joints_out = out['hand_joints_out'][0]
            obj_verts_out = out['obj_verts_refine']
            obj_verts_gt = torch.bmm(targets['R'].cuda(), out['obj_verts_template'].transpose(1,2)).transpose(1,2) + targets['T'].unsqueeze(1).cuda()

            # calculate hand metrics
            joint_dist = (hand_joints_out - targets['fit_joint_cam'].cuda()).norm(2, -1).mean(-1)
            hand_mje.append(joint_dist.item())

            bbox_pred = get_boundingbox_corners(obj_verts_out[0].cuda(), out['obj_faces'].cuda())
            bbox_gt = get_boundingbox_corners(obj_verts_gt[0].cuda(), out['obj_faces'].cuda())
            cpe = (bbox_pred - bbox_gt.float()).norm(2,-1).mean() 
            verts_dist.append(cpe.item())

            # calculate interaction metric
            pd, _ = get_inter_metrics(
                hand_verts_out.unsqueeze(0).float(),
                obj_verts_out.float(),
                tester.model.module.face.transpose(0,1).unsqueeze(0),
                out['obj_faces'].unsqueeze(0).cuda()
            )
            mean_penetration_depth.append(pd)

    if save_records:
        print("Hand MJE (cm):", (sum(hand_mje) / len(hand_mje)) * 100)
        print("Object MCE (cm):", (sum(verts_dist) / len(verts_dist)) * 100)
        print("PD (mm):", (sum(mean_penetration_depth) / len(mean_penetration_depth)) * 1000)
        print("Avg FPS:", (sum(avg_fps) / len(avg_fps)))

if __name__ == "__main__":
    args = parse_args()
    main()