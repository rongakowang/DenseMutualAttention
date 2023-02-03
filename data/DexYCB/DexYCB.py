import os
import pickle

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as TT
from libyana.meshutils import meshio
from manopth import manolayer
from pytorch3d.io import load_obj
from pytorch3d.transforms import matrix_to_quaternion
from utils.mano import MANO
from utils.preprocessing import augmentation, crop_image, generate_patch_image
from utils.transforms import *

import data.DexYCB.process_to_ho3d_pose as process_to_ho3d_pose
from data.HO3D.datasets import ho3dfullutils, manoutils
from data.HO3D.datasets.chunkvids import chunk_vid_index

filter_index = []

class DexYCB(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        root='../local_data/', 
        joint_nb=21,
        use_cache=False,
        mano_root=cfg.mano_root,
        mode="frame",
        ref_idx=0,
        valid_step=-1,
        frame_nb=10,
        track=False,
        chunk_size=1,
        chunk_step=4,
        box_mode="gt",
        ycb_root=cfg.ycb_dex_root,
        load_img=True,
        resize_scale=(256/640, 256/480),
        is_filtering=False
    ):
        super().__init__()

        self.name = "dex-ycb"
        self.mode = mode
        self.mano = MANO()
        self.vertex_num = self.mano.vertex_num
        self.joint_num = self.mano.joint_num
        self.transform = torchvision.transforms.ToTensor()
        self.resize_scale = resize_scale
        self.box_mode = box_mode
        self.root_joint_idx = self.mano.root_joint_idx
        self.is_filtering = is_filtering

        self.load_img = load_img
        self.frame_nb = frame_nb
        self.image_size = (256, 256)
        self.track = track
        # TODO: check hand setup
        self.setup = {"right_hand": 1, "objects": 1}
        cache_folder = os.path.join("../data/DexYCB", "cache")
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, f"{self.name}_{split}.pkl")

        self.root = os.path.join(root, self.name)
        if not os.path.exists(self.root):
            raise RuntimeError(
                f"DexYCB dataset not found at {self.root}")
        self.joint_nb = joint_nb
        # check joint order, this reorders to the canonical order
        self.reorder_idxs = np.array([
            0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8,
            9, 20
        ])

        # Fix dataset split
        valid_splits = ["train", "trainval", "val", "test"]
        assert split in valid_splits, "{} not in {}".format(
            split, valid_splits)
        self.split = split
        # check axis, this changes from openGL coordinate to canonical coordinate
        self.camextr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                 [0, 0, 0, 1]])
        # check MANO
        self.layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root=mano_root,
            center_idx=None,
            flat_hand_mean=True,
        )
        self.ref_idx = ref_idx
        self.cent_layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root=mano_root,
            center_idx=ref_idx,
            flat_hand_mean=True,
        )
        self.ycb_root = ycb_root
        self.obj_meshes = self.load_objects(self.ycb_root)
        self.can_obj_models = self.obj_meshes
        self.obj_paths = ho3dfullutils.load_object_paths(self.ycb_root)
        self.hand_faces = self.layer.th_faces
        if self.split == "train":
            subfolder = "s0_train"
        elif self.split == "val":
            subfolder = "s0_val"
        elif self.split == "test":
            subfolder = "s0_test"
        self.subfolder = subfolder
        root_meta = root_meta='../local_data/dex-ycb/meta'

        use_cache = True

        if os.path.exists(cache_path) and use_cache:
            
            with open(cache_path, "rb") as p_f:
                dataset_annots = pickle.load(p_f)
            frame_index = dataset_annots["frame_index"]
            annotations = dataset_annots["annotations"]
        else:
            frame_index, annotations = process_to_ho3d_pose.build_frame_index(
                root_meta, subfolder, use_cache=True)
            dataset_annots = {
                "frame_index": frame_index,
                "annotations": annotations,
            }
            with open(cache_path, "wb") as p_f:
                pickle.dump(dataset_annots, p_f)

        self.frame_index = frame_index
        self.annotations = annotations
        self.vid_index = self.frame_index.groupby(
            'seq_idx').first().reset_index()

    def load_objects(self, obj_root):
        import pytorch3d
        object_names = [
            obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name
        ]
        objects = {}
        
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, obj_name, "textured_simple_2000.obj")
            with open(obj_path) as m_f:
                mesh = meshio.fast_load_obj(m_f)[0]
            objects[obj_name] = {
                "verts": np.unique(mesh["vertices"], axis=0), 
                "faces": mesh["faces"],
                "path": obj_path
            }

            # NOTE: has to be read by pytorch3d in order to properly render 
            if objects[obj_name]['verts'].shape[0] != 1000 or 'meshlab' in obj_root:
                verts, faces, aux = load_obj(obj_path)
                objects[obj_name] = {
                    "verts": verts.cpu().numpy(),
                    "faces": faces.verts_idx.cpu().numpy(),
                    "path": obj_path,
                }
            
            assert objects[obj_name]['verts'].shape[0] == 1000

        return objects

    def project(self, points3d, cam_intr, camextr=None):
        if camextr is not None:
            points3d = np.array(self.camextr[:3, :3]).dot(
                points3d.transpose()).transpose()
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)

    def __len__(self):
        if self.mode == "frame":
            return len(self.frame_index)
        elif self.mode == "vid":
            return len(self.vid_index)
        elif self.mode == "chunk":
            return len(self.chunk_index)
        else:
            raise ValueError(f"{self.mode} mode not in [frame|vid|chunk]")


    def __getitem__(self, idx):
        if self.mode == "frame":
            row = self.frame_index.iloc[idx]
            frame_idx = row.frame_idx
            inputs, targets, meta_info = self.get_Feri_test(row, frame_idx)
            return inputs, targets, meta_info

    def get_image_path(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        img_path = os.path.join('../local_data/', annot["img"][2:])
        return img_path

    def load_image(self, seq_idx, frame_idx):
        img_path = self.get_image_path(seq_idx, frame_idx)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = img.astype(np.float32)
        return img

    def get_camintr(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        cam_intr = annot["camMat"]
        return cam_intr.copy()

    def name2index(self, obj_name, ycb_root):
        objects = sorted(os.listdir(ycb_root))
        return objects.index(obj_name)

    def get_obj_id(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot['objDir'].split('/')[3]
        return obj_id

    def get_obj_pose(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_pose = annot['objRt']
        rot = obj_pose[:3, :3]
        trans = obj_pose[:, 3:].reshape(3,)

        return rot, trans

    def get_obj_verts_trans(self, seq_idx, frame_idx):
        rot, trans = self.get_obj_pose(seq_idx, frame_idx)
        obj_id = self.get_obj_id(seq_idx, frame_idx)
        verts = self.obj_meshes[obj_id]["verts"]
        trans_verts = rot.dot(verts.transpose()).transpose() + trans
        obj_verts = np.array(trans_verts).astype(np.float32)
        return obj_verts

    def get_hand_ref(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        handpose = annot["mano_pose"][0]
        hand_trans = annot["hand_trans"][0]
        hand_shape = annot["mano_betas"]
        return handpose, hand_trans, hand_shape

    def get_hand_verts3d(self, seq_idx, frame_idx):
        handpose, hand_trans, hand_shape = self.get_hand_ref(
        seq_idx, frame_idx)
        handpose_th = torch.Tensor(handpose).unsqueeze(0)
        hand_joint_rots = handpose_th[:, self.cent_layer.rot:]
        hand_root_rot = handpose_th[:, :self.cent_layer.rot]
        hand_pca = manoutils.pca_from_aa(hand_joint_rots, rem_mean=True)
        handverts, handjoints, center_c = self.cent_layer(
            handpose_th,
            torch.Tensor(hand_shape).unsqueeze(0))
        hand_trans = hand_trans
        if center_c is not None:
            hand_trans = hand_trans + center_c.numpy()[0] / 1000
        handverts = handverts[0].numpy() / 1000 + hand_trans
        handjoints = handjoints[0].numpy() / 1000 + hand_trans
        # handverts = np.array(self.camextr[:3, :3]).dot(
        #         handverts.transpose()).transpose()
        # handjoints = np.array(self.camextr[:3, :3]).dot(
        #         handjoints.transpose()).transpose()

        return handverts, handjoints

    def crop_image(self, img, bbox, cam_param):
        from common.utils.preprocessing import generate_patch_image
        scale, rot, do_flip = 1.0, 0.0, False
        img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)
        cam_param_aug = trans @ cam_param
        cam_param_aug = np.concatenate([cam_param_aug, np.array([[0., 0., 1.]])], axis=0)
        return img, cam_param_aug

    def get_obj_verts_can(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot['objDir'].split('/')[3]
        verts = self.obj_meshes[obj_id]["verts"]
        return (verts - verts.mean(0)).astype(np.float32)

    def filter_out(self, joint_verts, obj_verts, cam_param):
        # remove instances that are: 
        # 1) hand or object not visible
        # 2) minimal distance between hand and object vertices are larger than 1cm
        hom_2d = (cam_param @ joint_verts.T).T
        hand_2d = (hom_2d / hom_2d[..., 2:])[..., :2]
        if hand_2d.min() < 0 or hand_2d.max() > 256:
            return True

        hom_2d = (cam_param @ obj_verts.T).T
        obj_2d = (hom_2d / hom_2d[..., 2:])[..., :2]
        if obj_2d.min() < 0 or obj_2d.max() > 256:
            return True

        for i in joint_verts:
            min_dist = (np.square(i - obj_verts).sum(axis=1) ** 0.5).min()
            if min_dist <= 0.01:
                return False

        return True

    def get_Feri_bbox(self, hand_verts, obj_verts, cam_param):
        "HO3D_bbox: [topLeftX, topLeftY, bottomRightX, bottomRightY]"
        "Feri_bbox: [topLeftX, topLeftY, width, height]"
        hom_2d = (cam_param @ hand_verts.T).T
        hand_2d = (hom_2d / hom_2d[..., 2:])[..., :2]
        hand_bbox = [hand_2d.min(axis=0)[0],hand_2d.min(axis=0)[1],hand_2d.max(axis=0)[0],hand_2d.max(axis=0)[1]]

        hom_2d = (cam_param @ obj_verts.T).T
        obj_2d = (hom_2d / hom_2d[..., 2:])[..., :2]
        obj_bbox = [obj_2d.min(axis=0)[0],obj_2d.min(axis=0)[1],obj_2d.max(axis=0)[0],obj_2d.max(axis=0)[1]]

        bbox = [min(hand_bbox[0], obj_bbox[0]), min(hand_bbox[1], obj_bbox[1]),
                max(hand_bbox[2], obj_bbox[2]), max((hand_bbox[3], obj_bbox[3]))]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        assert width > 0 and height > 0
        topLeftX = bbox[0]
        topLeftY = bbox[1]
        all_bbox = [topLeftX, topLeftY, width, height]
        square = max(all_bbox[2], all_bbox[3])
        all_bbox[2] = max(64, all_bbox[2], square)
        all_bbox[3] = max(64, all_bbox[3], square)
        all_bbox[0] = max(0, all_bbox[0])
        all_bbox[1] = max(0, all_bbox[1])
        if all_bbox[0] + all_bbox[2] > 256:
            all_bbox[0] = 256 - all_bbox[2]
        if all_bbox[1] + all_bbox[3] > 256:
            all_bbox[1] = 256 - all_bbox[3]
        return all_bbox, hand_bbox, obj_bbox

    def get_Feri_test(self, row, frame_idx):
        seq_idx = row.seq_idx
        img = self.load_image(seq_idx, frame_idx)
        img = cv2.resize(img, (256, 256))
        img_copy = img.copy()
        img_path = self.get_image_path(seq_idx, frame_idx)
        cam_param = self.get_camintr(seq_idx, frame_idx)
        if self.resize_scale is not None:
            cam_param[0, :] = cam_param[0,: ] * self.resize_scale[0]
            cam_param[1, :] = cam_param[1,: ] * self.resize_scale[1]


        mano_mesh_cam, mano_joint_cam = self.get_hand_verts3d(seq_idx, frame_idx)
        hand_root = mano_joint_cam[0, :].copy()
        obj_mesh_cam = self.get_obj_verts_trans(seq_idx, frame_idx)
        
        if cfg.bbox_crop:
            bbox, hand_bbox, obj_bbox = self.get_Feri_bbox(mano_mesh_cam, obj_mesh_cam, cam_param)
            img_check = img.copy()
            img_check = cv2.circle(img_check, (int(bbox[0]), int(bbox[1])), radius=0, color=(0, 0, 255), thickness=5)
            img_check = cv2.circle(img_check, (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), radius=0, color=(0, 0, 255), thickness=5)
            img, cam_param_aug = self.crop_image(img,
                                                 bbox,
                                                 cam_param)

        img = self.transform(img.astype(np.float32)) / 255.
        
        obj_center = obj_mesh_cam.mean(axis=0)

        obj_center_depth = obj_center[2]

        obj_name = self.get_obj_id(seq_idx, frame_idx)
        obj_class = self.name2index(obj_name, self.ycb_root)

        # if self.is_filtering:
        #     global filter_index
        #     if self.filter_out(mano_joint_cam, obj_mesh_cam, cam_param):
        #         filter_index.append((seq_idx, frame_idx))

        if cfg.bbox_crop:
            K = torch.Tensor(cam_param_aug)
        else:
            K = torch.Tensor(cam_param)
        R, T = self.get_obj_pose(seq_idx, frame_idx)
        R = torch.Tensor(R)
        T = torch.Tensor(T)

        R_vector = matrix_to_quaternion(torch.Tensor(R).unsqueeze(0)).squeeze(0)

        inputs = {'img': img, 'img_path': img_path}

        targets = {'fit_joint_cam': mano_joint_cam, 'fit_mesh_cam': mano_mesh_cam,
        'root_joint_depth': hand_root[2], 'obj_center_depth': obj_center_depth, 'obj_class': obj_class,
        'R':R, 'T':T, 'R_vector':R_vector, 'obj_center':obj_center,  'obj_verts': obj_mesh_cam, 'root_joint': hand_root} 

        meta_info = {'camera_intr': K,
         'focal': torch.Tensor([K[0,0], K[1,1]]), 'principal_point':torch.Tensor([K[0,2], K[1,2]]), 'data':1}
        
        return inputs, targets, meta_info