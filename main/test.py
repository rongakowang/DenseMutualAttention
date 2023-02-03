import argparse
import os
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from tqdm import tqdm

from common.base import Tester
from common.utils.ho3deval import dump
from common.utils.pointmertic import get_point_metrics

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

def get_inter_metrics(verts_person, verts_object, faces_person, faces_object):
    import common.utils.scenesdf as scenesdf
    sdfl = scenesdf.SDFSceneLoss([faces_person[0], faces_object[0]])
    _, sdf_meta = sdfl([verts_person, verts_object])
    # Mean penetration depth
    max_depths = sdf_meta['dist_values'][(1, 0)].mean(1)[0]
    # Valid contact
    has_contact = (max_depths > 0)

    return max_depths.item(), has_contact

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids, test_set='HO3D')
    cudnn.benchmark = True

    tester = Tester()
    tester._make_batch_generator()
    tester._make_model(args.model_path)

    sequences = ["SM1", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "AP10",
                "AP11", "AP12", "AP13", "AP14"]
                
    chamer_dist = []
    add_s = []
    verts_dists = []
    all_hands = []
    all_joints = []
    avg_fps = []
    mean_penetration_depth = []
    contacts = []
    pred_joints = {}
    pred_hands = {}
    for seq in sequences:
        pred_joints[seq] = []
        pred_hands[seq] = []

    save_records = True

    camextr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                 [0, 0, 0, 1]])

    for _, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        with torch.no_grad():
            obj_class = targets['obj_class'].item()
            
            start = time.time()
            out = tester.model(inputs, targets, meta_info, 'test')
            end = time.time()
            avg_fps.append((1/(end - start))) # calculate FPS

            hand_verts_out = out['hand_verts_out'][0]
            hand_joints_out = out['hand_joints_out'][0]
            obj_verts_out = out['obj_verts_refine']
            obj_verts_gt = torch.bmm(targets['R'].cuda(), out['obj_verts_template'].transpose(1,2)).transpose(1,2) + targets['T'].unsqueeze(1).cuda()

            # reorder to HO3D
            hand_joints_out = hand_joints_out[[0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20], :]

            # save to HO3D pred
            seq_idx = inputs['img_path'][0].split('/')[4]
            pred_hands[seq_idx].append(hand_verts_out.cpu().numpy().dot(camextr[:3, :3]))
            pred_joints[seq_idx].append(hand_joints_out.cpu().numpy().dot(camextr[:3, :3]))

            # calculate obj metrics
            obj_metrics = get_point_metrics(
                        obj_verts_out.float(),
                        obj_verts_gt.float())

            verts_dists.append(obj_metrics['verts_dists'][0])
            chamer_dist.append(obj_metrics['chamfer_dists'][0])
            add_s.append(obj_metrics['add-s'][0])

            # calculate interaction metric
            pd, has_contact = get_inter_metrics(
                hand_verts_out.unsqueeze(0).float(),
                obj_verts_out.float(),
                tester.model.module.face.transpose(0,1).unsqueeze(0),
                out['obj_faces'].unsqueeze(0).cuda()
            )
            mean_penetration_depth.append(pd)
            contacts.append(has_contact.item())

    if save_records:
        for seq in sequences:
            for k in range(len(pred_hands[seq])):
                all_hands.append(pred_hands[seq][k])
                all_joints.append(pred_joints[seq][k])

        print("Object MME (cm):", (sum(verts_dists) / len(verts_dists) * 100))
        print("Object ADD-S (cm):", (sum(add_s) / len(add_s)) * 100)
        print("PD (mm):", (sum(mean_penetration_depth) / len(mean_penetration_depth)) * 1000)
        print("CP (%):", (sum(contacts) / len(contacts)) * 100)
        print("Avg FPS:", (sum(avg_fps) / len(avg_fps)))

        os.makedirs('../ho3d_preds/', exist_ok=True)
        dump(f'../ho3d_preds/pred.json', all_joints, all_hands)

if __name__ == "__main__":
    args = parse_args()
    main()