import json
import shutil
import subprocess


def dump(pred_out_path, xyz_pred_list, verts_pred_list, codalab=True):
    """ Save predictions into a json file for official ho3dv2 evaluation. """

    xyz_pred_list = [x.round(4).tolist() for x in xyz_pred_list]
    verts_pred_list = [x.round(4).tolist() for x in verts_pred_list]

    # save to a json
    print(f"Dumping json results to {pred_out_path}")
    with open(pred_out_path, "w") as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print("Dumped %d joints and %d verts predictions to %s" %
          (len(xyz_pred_list), len(verts_pred_list), pred_out_path))
    if codalab:
        save_zip_path = pred_out_path.replace(".json", ".zip")
        subprocess.call(["zip", "-j", save_zip_path, pred_out_path])
        print(f"Saved results to {save_zip_path}")