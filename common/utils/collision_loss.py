import numpy as np
import torch
import torch.nn as nn


import common.utils.contactloss as contactloss
import common.utils.scenesdf as scenesdf


def tensorify(array, device=None):
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array)
    if device is not None:
        array = array.to(device)
    return array

def compute_collision_loss(verts_hand,
                           verts_object,
                           mano_faces,
                           faces_object,
                           collision_mode="sdf"):
    if collision_mode == "sdf":
        # mano_faces = faces_object.new(MANO_CLOSED_FACES)
        mano_faces = faces_object
        sdfl = scenesdf.SDFSceneLoss([mano_faces, faces_object])
        sdf_loss, sdf_meta = sdfl([verts_hand, verts_object])
        return sdf_loss.mean()

def compute_contact_loss(verts_hand_b, verts_object_b, faces_hand_closed, faces_object):
    missed_loss, contact_loss, _, _ = contactloss.compute_contact_loss(
            verts_hand_b, faces_hand_closed, verts_object_b, faces_object)
    return missed_loss + contact_loss