import os

import neural_renderer as nr
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from libyana.meshutils import meshio
from pytorch3d.io import load_obj
from pytorch3d.transforms import quaternion_to_matrix
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import FaceToEdge

from common.nets.layer import make_conv1d_layers
from common.nets.loss import (CoordLoss, EdgeLengthLoss, NormalVectorLoss,
                              ParamLoss)
from common.nets.module_no_render import (GCN_A, MeshNet, PoseNet,
                                          SegmentationNet)
from common.nets.resnet import ResNetBackbone
from common.utils.mano import MANO
from common.utils.transforms import pixel2cam


class Model(nn.Module):
    def __init__(self, hand_img_encoder, pose_net, hand_mesh_net, segmentation_net, obj_img_encoder, obj_mesh_net):
        super(Model, self).__init__()
        self.pose_backbone = hand_img_encoder
        self.pose_net = pose_net
        self.mesh_net = hand_mesh_net
        self.gcn_a = GCN_A(input_dim=3 + 64 + 64 + 256 + 512 + 3, hidden_dim=64, output_dim_h=3, output_dim_o=7, drop_edge=True)
        self.segmentation_net = segmentation_net
        self.global_img_feat = nn.Sequential(
            make_conv1d_layers([2048, 512], kernel=1, stride=1, padding=0, bnrelu_final=False)
        )

        self.human_model = MANO()
        self.human_model_layer = self.human_model.layer.to(torch.cuda.current_device())

        self.root_joint_idx = self.human_model.root_joint_idx
        self.joint_regressor = self.human_model.joint_regressor

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.normal_loss = NormalVectorLoss(self.human_model.face)
        self.edge_loss = EdgeLengthLoss(self.human_model.face)
        self.mask_loss = nn.CrossEntropyLoss()
        self.face = self.human_model_layer.th_faces.permute(1, 0).to(torch.cuda.current_device())
        self.face2edge = FaceToEdge()

        self.renderer = nr.renderer.Renderer(image_size=256)
        self.templates = self.prepare_model_template(cfg.ycb_ho3d_root)
        self.dex_templates = self.prepare_model_template(cfg.ycb_dex_root)

        self.renderer.light_intensity_direction = 0
        self.renderer.light_intensity_ambient = 1

        self.mask_backbone = obj_img_encoder
        self.obj_mesh_net = obj_mesh_net


    def prepare_model_template(self, obj_root):
        templates = [] # faces order depends on the os.listdir
        for obj in sorted(os.listdir(obj_root)):
            path = os.path.join(obj_root, obj, 'textured_simple_2000.obj')
            with open(path) as m_f:
                mesh = meshio.fast_load_obj(m_f)[0]
                if mesh['vertices'].shape[0] != 1000 or 'meshlab' in obj_root:
                    verts, faces, aux = load_obj(path)
                    assert verts.shape[0] == 1000
                    templates.append({
                        "verts": verts,
                        "face": faces.verts_idx.long(),
                    })
                else:
                    templates.append({"verts": torch.Tensor(mesh["vertices"]), "face": torch.Tensor(mesh["faces"]).long()})
        return templates

    def project(self, img_feat, vertices):
        v = vertices[:, :, :2]
        v = v.unsqueeze(2)
        v = v / 32. - 1.0
        output = F.grid_sample(img_feat, v, align_corners=False)
        return output.squeeze(-1).permute(0, 2, 1)

    def forward(self, inputs, targets, meta_info, mode):

        #############################
        # Initial Stage
        #############################

        shared_img_feat, pose_img_feat, feats, feats128 = self.pose_backbone(inputs['img'])
        oshared_img_feat, mask_img_feat, ofeats, ofeats128 = self.mask_backbone(inputs['img'])

        if mode == 'train':
            mask = self.segmentation_net(mask_img_feat, ofeats, ofeats128)
            joint_coord_img, _, _ = self.pose_net(pose_img_feat)

        mesh_coord_img = self.mesh_net(pose_img_feat)
        obj_rough_img = self.obj_mesh_net(mask_img_feat)
        rough_mesh = torch.cat([mesh_coord_img, obj_rough_img], dim=1)

        #############################
        # Refinement Stage
        #############################

        class_id = targets['obj_class'] 
        templates = []
        for i, idx in enumerate(class_id):
            if meta_info['data'][i] == 0:
                templates.append(self.templates[idx])
            else:
                templates.append(self.dex_templates[idx])
        
        obj_verts_template = torch.stack([obj["verts"] for obj in templates],dim=0).cuda()
        obj_face =  templates[0]['face']
        obj_faces_pad_gcn = torch.stack([F.pad(obj["face"], (0,0,0,2007 - obj["face"].shape[0]), mode='constant', value=0) 
                                        for obj in templates],dim=0).cuda() # bs * 2007 * 3

        B_h = Batch.from_data_list([Data(x=mesh_coord_img[b], face=self.face).to(torch.cuda.current_device()) for b in range(mesh_coord_img.shape[0])])
        B_o = Batch.from_data_list([Data(x=obj_rough_img[b], face=obj_faces_pad_gcn[b].T).to(torch.cuda.current_device()) for b in range(mesh_coord_img.shape[0])])
        B_h = self.face2edge(B_h)
        B_o = self.face2edge(B_o)

        global_img_feat = self.global_img_feat((pose_img_feat + mask_img_feat).mean((2,3))[:,:,None]).permute(0, 2, 1)

        proj_img_h = self.project(inputs['img'], B_h.x.view(-1, 778, 3))
        proj_feat0_h = self.project(feats128, B_h.x.view(-1, 778, 3))
        proj_feat1_h = self.project(shared_img_feat, B_h.x.view(-1, 778, 3))
        proj_feat2_h = self.project(feats[0], B_h.x.view(-1, 778, 3))
        cat_feat_h = torch.cat((proj_img_h, proj_feat0_h, proj_feat1_h, proj_feat2_h, global_img_feat.repeat(1, 778, 1)), dim=2).view(-1, 3 + 64 + 64 + 256 + 512)

        proj_img_o = self.project(inputs['img'], B_o.x.view(-1, 1000, 3))
        proj_feat0_o = self.project(ofeats128, B_o.x.view(-1, 1000, 3))
        proj_feat1_o = self.project(oshared_img_feat, B_o.x.view(-1, 1000, 3))
        proj_feat2_o = self.project(ofeats[0], B_o.x.view(-1, 1000, 3))
        cat_feat_o = torch.cat((proj_img_o, proj_feat0_o, proj_feat1_o, proj_feat2_o, global_img_feat.repeat(1, 1000, 1)), dim=2).view(-1, 3 + 64 + 64 + 256 + 512)

        B_h.x = torch.cat((B_h.x, cat_feat_h), dim=1)
        B_o.x = torch.cat((B_o.x, cat_feat_o), dim=1)
        x_h, x_o, _, _ = self.gcn_a(B_h, B_o)
        hand_x = (rough_mesh[:,:778,:] + x_h[:,:778,:3])
        R_vector_refine = x_o[...,:4].mean(dim=1)

        if mode == 'train':
            T_refine = x_o[...,4:].mean(dim=1) + targets['fit_joint_cam'][:,0,:] # relative to hand joint
        else:
            T_refine = x_o[...,4:].mean(dim=1) + targets['root_joint']

        fine_joint_img_from_mesh = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None, :, :].repeat(hand_x.shape[0], 1, 1), hand_x)

        obj_rot_refine = quaternion_to_matrix(R_vector_refine)
        obj_verts_refine = torch.bmm(obj_rot_refine, obj_verts_template.transpose(1,2)).transpose(1,2) + T_refine.unsqueeze(1)

        if mode == 'test': # saving for the rotation file
            hand_verts = torch.zeros_like(hand_x)
            hand_verts[...,0] = hand_x[...,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            hand_verts[...,1] = hand_x[...,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            hand_verts[...,2] = (hand_x[...,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2) + targets['root_joint_depth'].unsqueeze(1)
            
            hand_verts_new = torch.zeros_like(hand_x)
            for i in range(hand_verts.shape[0]):
                hand_verts_new[i] = pixel2cam(hand_verts[i], meta_info['focal'][i], meta_info['principal_point'][i])

            hand_joints = torch.zeros_like(fine_joint_img_from_mesh)
            hand_joints[...,0] = fine_joint_img_from_mesh[...,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            hand_joints[...,1] = fine_joint_img_from_mesh[...,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            hand_joints[...,2] = (fine_joint_img_from_mesh[...,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2) + targets['root_joint_depth'].unsqueeze(1)
            hand_joints_new = torch.zeros_like(hand_joints)
            for i in range(hand_joints.shape[0]):
                hand_joints_new[i] = pixel2cam(hand_joints[i], meta_info['focal'][i], meta_info['principal_point'][i])

        out =  {'obj_verts_refine': obj_verts_refine,
                'hand_verts_out': hand_verts_new,
                'hand_joints_out': hand_joints_new,
                'obj_verts_template': obj_verts_template,
                'obj_faces': obj_face} 
        return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight)
    elif type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias,0)

def get_model(vertex_num, joint_num, mode):
    hand_img_encoder = ResNetBackbone(cfg.resnet_type)
    pose_net = PoseNet(joint_num)
    hand_mesh_net = MeshNet(vertex_num)
    
    obj_img_encoder = ResNetBackbone(cfg.resnet_type, input_dim=3)
    segmentation_net = SegmentationNet(out=2) # predicting for obj mask only
    obj_mesh_net = MeshNet(1000)

    if mode == 'train':
        hand_img_encoder.init_weights()
        pose_net.apply(init_weights)
        hand_mesh_net.apply(init_weights)

        obj_img_encoder.init_weights(skip_first=True)
        segmentation_net.apply(init_weights)
        obj_mesh_net.apply(init_weights)

    model = Model(hand_img_encoder, pose_net, hand_mesh_net, segmentation_net, obj_img_encoder, obj_mesh_net)
    return model



