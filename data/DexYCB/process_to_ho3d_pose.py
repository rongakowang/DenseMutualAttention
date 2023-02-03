# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

import os
import pickle

import numpy as np
import pandas as pd
import torch
from dex_ycb_toolkit.factory import get_dataset
from manopth.manolayer import ManoLayer
from tqdm import tqdm


def build_frame_index(root_meta='../local_data/dex-ycb/meta', subfolder="s0_train", use_cache=True, cache_folder="../data/DexYCB/cache"):
    """
    root_meta: root of meta pkl file
    subfolder: one of  's0_train', 's0_val', 's0_test'
    """

    seqs = []
    for sub in os.listdir(os.path.join(root_meta,subfolder)):
        for sub1 in os.listdir(os.path.join(root_meta,subfolder,sub)):
            for sub2 in os.listdir(os.path.join(root_meta,subfolder,sub,sub1)):
                seqs.append(os.path.join(sub,sub1,sub2))

    cache_path = os.path.join(cache_folder, f"{subfolder}.pkl")
    if not os.path.exists(cache_path):
        os.makedirs(cache_folder, exist_ok=True)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as p_f:
            pd_frame_index, annotations_map = pickle.load(p_f)
    else:
        with open(f'../local_data/dex-ycb/meta/Dex_{subfolder[3:]}_filter_1cm.pth', 'rb') as f:
          filter_index = pickle.load(f)
        annotations_map = {}
        annotations_list = []
        seq_lengths = {}
        # for seq in tqdm(sorted(seqs), desc="seq"):
        scene_id = 0
        for seq in tqdm(sorted(seqs)):
            # seq_folder = os.path.join(root_original, seq)
            meta_folder = os.path.join(root_meta,subfolder, seq)
            # rgb_folder = seq_folder

            img_nb = len(os.listdir(meta_folder))
            seq_lengths[seq] = img_nb
            # for frame_idx in tqdm(range(img_nb), desc="img"):
            for frame_idx in range(img_nb):
                meta_path = os.path.join(meta_folder, f"meta_{frame_idx:06d}.pkl")
                with open(meta_path, "rb") as p_f:
                    annot = pickle.load(p_f)
                # img_path = os.path.join(rgb_folder, f"color_{frame_idx:06d}.jpg")
                # annot["img"] = img_path
                annot["seq_idx"] = seq
                annot["frame_idx"] = frame_idx
                annot["frame_nb"] = img_nb
                # annot["scene_id"] = scene_id
                # annot["view_id"] = 0
                assert len(annot['mano_pose']) == 1 
                #  and (seq, frame_idx) not in filter_index
                if not (annot['mano_pose'] == 0).all() and annot['mano_side'] == 'right' and (seq, frame_idx) not in filter_index:
                  scene_id += 1
                  annotations_map[(seq, frame_idx)] = annot
                  annotations_list.append(annot)
        pd_frame_index = pd.DataFrame(
                annotations_list).loc[:, ["img", "seq_idx", "frame_idx", "frame_nb"]]
        with open(cache_path, "wb") as p_f:
            pickle.dump((pd_frame_index, annotations_map), p_f)
    return pd_frame_index, annotations_map



def create_scene(sample, obj_file, name):
  """Creates the pyrender scene of an image sample.

  Args:
    sample: A dictionary holding an image sample.
    obj_file: A dictionary holding the paths to YCB OBJ files.

  Returns:
    A pyrender scene object.
  """

  ret = {}

  # Add camera.
  fx = sample['intrinsics']['fx']
  fy = sample['intrinsics']['fy']
  cx = sample['intrinsics']['ppx']
  cy = sample['intrinsics']['ppy']

  cam = np.eye(3)
  cam[0,0] = fx
  cam[1,1] = fy
  cam[0,-1] = cx
  cam[1,-1] = cy

  ret['camMat'] = cam


  # Load poses.
  label = np.load(sample['label_file'])
  pose_y = label['pose_y']
  pose_m = label['pose_m']

  # Load YCB meshes.
  i = sample['ycb_ids'][sample['ycb_grasp_ind']]
  obj_dir = obj_file[i]
  obj_pose = pose_y[sample['ycb_grasp_ind']]

  ret['objDir'] = obj_dir.replace('/home/wei/Documents/1T/','./')
  ret['objRt'] = obj_pose
  ret['img'] = sample['color_file'].replace('/home/wei/Documents/1T/','./')
  ret['mano_betas'] = sample['mano_betas']
  ret['mano_side'] = sample['mano_side']

  # Load MANO layer.
  mano_layer = ManoLayer(flat_hand_mean=False,
                         ncomps=45,
                         side=sample['mano_side'],
                         mano_root='/home/wei/Documents/projects/2022-hand-object/dex-ycb-toolkit/manopth/mano/models',
                         use_pca=True)
  mano_layer_ho3d = ManoLayer(
                              joint_rot_mode="axisang",
                              use_pca=False,
                              mano_root='/home/wei/Documents/projects/2022-hand-object/dex-ycb-toolkit/manopth/mano/models',
                              center_idx=None,
                              flat_hand_mean=True,
                              side=sample['mano_side'],
                              )
  faces = mano_layer.th_faces.numpy()
  betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)

  # Add MANO meshes.
  if not np.all(pose_m == 0.0):
    pose = torch.from_numpy(pose_m)
    trans = pose[:, 48:51]
    pose = pose[:, 0:48]
    ret['hand_trans'] = trans.cpu().data.numpy()
    th_hand_pose_coeffs = pose[:, mano_layer.rot:mano_layer.rot +mano_layer.ncomps]
    th_full_hand_pose = th_hand_pose_coeffs.mm(mano_layer.th_selected_comps)
    # Concatenate back global rot
    th_full_pose = torch.cat([
      pose[:, :mano_layer.rot],
      mano_layer.th_hands_mean + th_full_hand_pose
    ], 1)
    ret['mano_pose'] = th_full_pose.cpu().data.numpy()
    if np.random.rand(1)[0] < 0.01:
      vert, _, _ = mano_layer(pose, betas, trans)
      vert_ho3d, _, _ = mano_layer_ho3d(th_full_pose,betas,trans)
      assert (vert_ho3d-vert).abs().max() < 0.0001
  else:
    ret['mano_pose'] = np.zeros([1,48])
    ret['hand_trans'] = np.zeros([1,3])

  ret_dir = os.path.join(f'/home/wei/Documents/dex-ycb/{name}',*(ret['img'].split('/')[2:-1]))
  if not os.path.exists(ret_dir):
    os.makedirs(ret_dir)
  frame_id = int(ret['img'].split('/')[-1][:-4].split('_')[-1])
  ret_file = ret_dir + f'/meta_{frame_id:06d}.pkl'
  with open(ret_file, "wb") as p_f:
    pickle.dump(ret, p_f)

def main():
  names = ['s0_train','s0_val','s0_test']

  for name in names:
    dataset = get_dataset(name)

    for idx in tqdm(range(len(dataset))):
      # idx = 70
      sample = dataset[idx]

      create_scene(sample, dataset.obj_file, name)


if __name__ == '__main__':
  build_frame_index()
  # main()
