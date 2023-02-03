import os
import pickle

import cv2
import numpy as np
import torch
import torchvision
from libyana.meshutils import meshio
from manopth import manolayer
from PIL import Image
from pytorch3d.transforms import matrix_to_quaternion

from common.utils.mano import MANO
from common.utils.transforms import *
from data.HO3D.datasets import ho3dconstants, ho3dfullutils, ho3dutils


class HO3D(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        root='../local_data',
        joint_nb=21,
        use_cache=False,
        mano_root=cfg.mano_root,
        mode="frame",
        ref_idx=0,
        box_mode="gt",
        ycb_root=cfg.ycb_ho3d_root,
        load_img=True,
        resize_scale=(256/640, 256/480),
    ):
        super().__init__()

        self.name = "ho3d"
        self.mode = mode
        self.mano = MANO()
        self.vertex_num = self.mano.vertex_num
        self.joint_num = self.mano.joint_num
        self.transform = torchvision.transforms.ToTensor()
        self.resize_scale = resize_scale
        self.box_mode = box_mode
        self.root_joint_idx = self.mano.root_joint_idx
        self.ycb_root = ycb_root
        self.load_img = load_img
        self.image_size = (256, 256)
        self.setup = {"right_hand": 1, "objects": 1}
        cache_folder = os.path.join("../data/HO3D", "cache")
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, f"{self.name}_{split}.pkl")

        self.root = os.path.join(root, self.name)
        if not os.path.exists(self.root):
            raise RuntimeError(
                f"HO3D dataset not found at {self.root}, please follow instructions"
                "at https://github.com/hassony2/homan/tree/master#ho-3d")
        self.joint_nb = joint_nb
        self.reorder_idxs = np.array([
            0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8,
            9, 20
        ])

        # Fix dataset split
        valid_splits = ["train", "trainval", "val", "test"]
        assert split in valid_splits, "{} not in {}".format(
            split, valid_splits)
        self.split = split
        self.camextr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                 [0, 0, 0, 1]])
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
        self.obj_meshes = ho3dfullutils.load_objects(ycb_root)
        self.can_obj_models = self.obj_meshes
        self.obj_paths = ho3dfullutils.load_object_paths(ycb_root)
        self.hand_faces = self.layer.th_faces
        if self.split == "train":
            seqs = ho3dconstants.TRAIN_SEQS
            subfolder = "train"
        elif self.split == "trainval":
            seqs = ho3dconstants.TRAINVAL_SEQS
            subfolder = "train"
        elif self.split == "val":
            seqs = ho3dconstants.VAL_SEQS
            subfolder = "train"
        elif self.split == "test":
            seqs = ho3dconstants.TEST_SEQS
            subfolder = "evaluation"
        self.subfolder = subfolder

        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as p_f:
                dataset_annots = pickle.load(p_f)
            frame_index = dataset_annots["frame_index"]
            annotations = dataset_annots["annotations"]
        else:
            frame_index, annotations = ho3dutils.build_frame_index(
                seqs, self.root, subfolder, use_cache=use_cache)
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

        # Get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

        self.obj_templates = self.prepare_model_template(self.ycb_root)

    def get_root(self, seq_idx, frame_idx):
        hand_root = self.annotations[(seq_idx, frame_idx)]['handJoints3D']
        # hand_joints = hand_root[np.newaxis].repeat(21, 0)
        return hand_root.dot(self.camextr[:3, :3])

    def __getitem__(self, idx):
        if self.mode == "frame":
            row = self.frame_index.iloc[idx]
            frame_idx = row.frame_idx
            inputs, targets, meta_info = self.get_Feri_test(row, frame_idx)
            return inputs, targets, meta_info

    def pil2opencv(self, img):
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image

    def opencv2pil(self, img):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return pil_img

    def crop_image(self, img, bbox, cam_param):
        from common.utils.preprocessing import generate_patch_image
        scale, rot, do_flip = 1.0, 0.0, False
        img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)
        cam_param_aug = trans @ cam_param
        cam_param_aug = np.concatenate([cam_param_aug, np.array([[0., 0., 1.]])], axis=0)
        return img, cam_param_aug

    def get_Feri_test(self, row, frame_idx):
        seq_idx = row.seq_idx
        img = self.load_image(seq_idx, frame_idx)
        img = cv2.resize(img, cfg.input_img_shape)
        img_path = self.get_image_path(seq_idx, frame_idx)

        cam_param = self.get_camintr(seq_idx, frame_idx)
        if self.resize_scale is not None:
            cam_param[0, :] = cam_param[0,: ] * self.resize_scale[0]
            cam_param[1, :] = cam_param[1,: ] * self.resize_scale[1]

        if cfg.bbox_crop:
            bbox = self.get_Feri_bbox(seq_idx, frame_idx)
            img, cam_param_aug = self.crop_image(img,
                                                 bbox,
                                                 cam_param)

        img = self.transform(img.astype(np.float32)) / 255.
        mano_mesh_cam, mano_joint_cam = self.get_hand_verts3d(seq_idx, frame_idx)
        hand_root = self.get_root(seq_idx, frame_idx)
        obj_mesh_cam = self.get_obj_verts_trans(seq_idx, frame_idx)
        obj_name = self.get_obj_id(seq_idx, frame_idx)
        obj_class = self.name2index(obj_name, self.ycb_root)

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
        'root_joint_depth': hand_root[2], 'obj_center_depth': hand_root[2], 'obj_class': obj_class,
        'R':R, 'T':T, 'R_vector':R_vector, 'obj_center':hand_root, 'obj_verts': obj_mesh_cam, 'root_joint': hand_root} 

        meta_info = {'camera_intr': K,
         'focal': torch.Tensor([K[0,0], K[1,1]]), 'principal_point':torch.Tensor([K[0,2], K[1,2]]), 'data':0}
        
        return inputs, targets, meta_info

    def process_mask_image(self, mask):
        mask = self.transform(mask.astype(np.float32)) / 255.
        mask = mask[0, :, :]
        mask = np.round(mask).long()
        return mask

    def get_image_path(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        img_path = annot["img"]
        return img_path

    def get_image(self, seq_idx, frame_idx, load_rgb=True):
        img_path = self.get_image_path(seq_idx, frame_idx)
        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    def load_image(self, seq_idx, frame_idx):
        """
        default load order is BGR, will take care in preprocessing
        """
        img_path = self.get_image_path(seq_idx, frame_idx)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = img.astype(np.float32)
        return img

    def get_joints2d(self, seq_idx, frame_idx):
        joints3d = self.get_joints3d(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        return self.project(joints3d, cam_intr)

    def get_joints3d(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        joints3d = annot["handJoints3D"]
        joints3d = self.camextr[:3, :3].dot(joints3d.transpose()).transpose()
        if joints3d.ndim == 1:
            joints3d = joints3d[np.newaxis].repeat(21, 0)
        joints3d = joints3d[self.reorder_idxs]
        return joints3d.astype(np.float32)

    def get_obj_textures(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        textures = self.obj_meshes[obj_id]["textures"]
        return textures

    def get_hand_ref(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        # Retrieve hand info
        if "handPose" in annot:
            handpose = annot["handPose"]
            hand_trans = annot["handTrans"]
            hand_shape = annot["handBeta"]
            # print(handpose.shape, hand_trans.shape, hand_shape.shape, "I am here")
            # assert False
            trans = hand_trans
        else:
            handpose = np.zeros(48)
            trans = annot["handJoints3D"]
            hand_shape = np.zeros(10)
        return handpose, trans, hand_shape

    def get_hand_verts3d(self, seq_idx, frame_idx):
        handpose, hand_trans, hand_shape = self.get_hand_ref(
            seq_idx, frame_idx)
        handpose_th = torch.Tensor(handpose).unsqueeze(0)
        # hand_joint_rots = handpose_th[:, self.cent_layer.rot:]
        # hand_root_rot = handpose_th[:, :self.cent_layer.rot]
        # hand_pca = manoutils.pca_from_aa(hand_joint_rots, rem_mean=True)
        handverts, handjoints, center_c = self.cent_layer(
            handpose_th,
            torch.Tensor(hand_shape).unsqueeze(0))
        hand_trans = hand_trans
        if center_c is not None:
            hand_trans = hand_trans + center_c.numpy()[0] / 1000
        handverts = handverts[0].numpy() / 1000 + hand_trans
        handjoints = handjoints[0].numpy() / 1000 + hand_trans

        handverts = np.array(self.camextr[:3, :3]).dot(
                handverts.transpose()).transpose()
        handjoints = np.array(self.camextr[:3, :3]).dot(
                handjoints.transpose()).transpose()

        return handverts, handjoints

    def get_hand_verts2d(self, seq_idx, frame_idx):
        verts3d, _ = self.get_hand_verts3d(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        verts2d = self.project(verts3d, cam_intr, self.camextr)
        return verts2d

    def get_obj_path(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        obj_path = self.obj_paths[obj_id]
        return obj_path

    def get_obj_faces(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        objfaces = self.obj_meshes[obj_id]["faces"]
        objfaces = np.array(objfaces).astype(np.int16)
        return objfaces

    def get_obj_id(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        return obj_id

    def get_obj_rot(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        return rot

    def get_obj_verts_can(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        verts = self.obj_meshes[obj_id]["verts"]
        return (verts - verts.mean(0)).astype(np.float32)

    def get_obj_verts2d(self, seq_idx, frame_idx):
        verts3d = self.get_obj_verts_trans(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        verts2d = self.project(verts3d, cam_intr)
        return verts2d

    def get_obj_verts_trans(self, seq_idx, frame_idx):
        rot = self.get_obj_rot(seq_idx, frame_idx)
        annot = self.annotations[(seq_idx, frame_idx)]
        trans = annot["objTrans"]
        obj_id = annot["objName"]
        verts = self.obj_meshes[obj_id]["verts"]
        trans_verts = rot.dot(verts.transpose()).transpose() + trans
        trans_verts = self.camextr[:3, :3].dot(
            trans_verts.transpose()).transpose()
        obj_verts = np.array(trans_verts).astype(np.float32)
        return obj_verts

    def get_obj_pose(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        rot = self.get_obj_rot(seq_idx, frame_idx)
        trans = annot["objTrans"]
        rot = self.camextr[:3, :3].dot(rot)
        trans = self.camextr[:3, :3].dot(trans)

        return rot, trans

    def get_obj_corners3d(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        trans = annot["objTrans"]
        corners = annot["objCorners3DRest"]
        trans_corners = rot.dot(corners.transpose()).transpose() + trans
        trans_corners = self.camextr[:3, :3].dot(
            trans_corners.transpose()).transpose()
        obj_corners = np.array(trans_corners).astype(np.float32)
        return obj_corners

    def get_objcorners2d(self, seq_idx, frame_idx):
        corners3d = self.get_obj_corners3d(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        return self.project(corners3d, cam_intr)

    def get_objverts2d(self, seq_idx, frame_idx):
        objpoints3d = self.get_obj_verts_trans(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        verts2d = self.project(objpoints3d, cam_intr)
        return verts2d

    def get_sides(self, idx):
        return "right"

    def get_camintr(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        cam_intr = annot["camMat"]
        return cam_intr.copy()

    def get_focal_nc(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        cam_intr = annot["camMat"]
        return (cam_intr[0, 0] + cam_intr[1, 1]) / 2 / max(self.image_size)

    def __len__(self):
        if self.mode == "frame":
            return len(self.frame_index)
        elif self.mode == "vid":
            return len(self.vid_index)
        elif self.mode == "chunk":
            return len(self.chunk_index)
        else:
            raise ValueError(f"{self.mode} mode not in [frame|vid|chunk]")

    def project(self, points3d, cam_intr, camextr=None):
        if camextr is not None:
            points3d = np.array(self.camextr[:3, :3]).dot(
                points3d.transpose()).transpose()
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)

    def get_hand_bbox(self, seq_idx, frame_idx):
        
        if self.box_mode in ["track"]:
            bbox = np.array(
                self.tracked_boxes[seq_idx]['right_hand'][frame_idx])
            # bbox = np.array(self.tracked_boxes[seq_idx]['hands'][0][frame_idx])
        elif self.box_mode == "gt":
            annot = self.annotations[(seq_idx, frame_idx)]
            if "handBoundingBox" in annot:
                bbox = np.array(annot["handBoundingBox"])
                # warnings.warn(f"Problem with gt bboxes ?")
            else:
                verts3d, _ = self.get_hand_verts3d(seq_idx, frame_idx)
                cam_intr = self.get_camintr(seq_idx, frame_idx)
                cam2 = cam_intr.copy()
                if self.resize_scale is not None:
                    cam2[0, :] = cam2[0,: ] * self.resize_scale[0]
                    cam2[1, :] = cam2[1,: ] * self.resize_scale[1]
                verts2d = self.project(verts3d, cam2)
                bbox = np.concatenate([verts2d.min(0), verts2d.max(0)], 0)
        else:
            raise ValueError(
                f"Invalid box mode {self.box_mode}, not in ['track'|'gt']")
        return bbox

    def get_obj_bbox(self, seq_idx, frame_idx):
        if self.box_mode in ["track"]:
            bbox = np.array(self.tracked_boxes[seq_idx]['objects'][frame_idx])
        elif self.box_mode == "gt":
            verts3d = self.get_obj_verts_trans(seq_idx, frame_idx)
            cam_intr = self.get_camintr(seq_idx, frame_idx)
            cam2 = cam_intr.copy()
            if self.resize_scale is not None:
                cam2[0, :] = cam2[0,: ] * self.resize_scale[0]
                cam2[1, :] = cam2[1,: ] * self.resize_scale[1]
            verts2d = self.project(verts3d, cam2)
            # verts2d = self.get_obj_verts2d(seq_idx, frame_idx)
            bbox = np.concatenate([verts2d.min(0), verts2d.max(0)], 0)
        else:
            raise ValueError(
                f"Invalid box mode {self.box_mode}, not in ['track'|'gt']")
        return bbox

    def get_Feri_bbox(self, seq_idx, frame_idx):
        "HO3D_bbox: [topLeftX, topLeftY, bottomRightX, bottomRightY]"
        "Feri_bbox: [topLeftX, topLeftY, width, height]"
        hand_bbox = self.get_hand_bbox(seq_idx, frame_idx)
        obj_bbox = self.get_obj_bbox(seq_idx, frame_idx)
        "In test mode, the bbox are given by gt, need to resize"
        hand_bbox = [hand_bbox[0] * self.resize_scale[0], hand_bbox[1] * self.resize_scale[1],
                     hand_bbox[2] * self.resize_scale[0], hand_bbox[3] * self.resize_scale[1]]


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
        return all_bbox

    def name2index(self, obj_name, ycb_root):
        objects = sorted(os.listdir(ycb_root))
        return objects.index(obj_name)

    def compute_vertex(self, mask, kpt_2d):
        h, w = mask.shape
        m = kpt_2d.shape[0]
        xy = np.argwhere(mask == 1)[:, [1, 0]]

        vertex = kpt_2d[None] - xy[:, None]
        norm = np.linalg.norm(vertex, axis=2, keepdims=True)
        norm[norm < 1e-3] += 1e-3
        vertex = vertex / norm

        vertex_out = np.zeros([h, w, m, 2], np.float32)
        vertex_out[xy[:, 1], xy[:, 0]] = vertex
        vertex_out = np.reshape(vertex_out, [h, w, m * 2])

        return vertex_out

    def prepare_model_template(self, obj_root):
        templates = [] # faces order depends on the os.listdir
        for obj in sorted(os.listdir(obj_root)):
            path = os.path.join(obj_root, obj, 'textured_simple_2000.obj')
            with open(path) as m_f:
                mesh = meshio.fast_load_obj(m_f)[0]
                templates.append({"verts": torch.Tensor(mesh["vertices"]), "face": torch.Tensor(mesh["faces"]).long()})
        return templates