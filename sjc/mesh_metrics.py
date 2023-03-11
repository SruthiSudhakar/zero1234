import point_cloud_utils as pcu
import trimesh
import numpy as np
from trimesh.voxel.creation import voxelize, local_voxelize


def sample_surf_points(v, f, num_samples):
    f_i, bc = pcu.sample_mesh_random(v, f, num_samples=num_samples)
    # Use the face indices and barycentric coordinate to compute sample positions and normals
    v_sampled = pcu.interpolate_barycentric_coords(f, f_i, bc, v)
    return v_sampled


def scale_to_unit_cube(v):
    aabb_max = v.max(axis=0)
    aabb_min = v.min(axis=0)
    scale = 0.5 / (aabb_max - aabb_min).max()
    v = v * scale
    return v


def compute_iou(vox1, vox2):
    # compute iou
    vox1 = vox1.astype(bool)
    vox2 = vox2.astype(bool)
    iou = (vox1 & vox2).sum() / (vox1 | vox2).sum()
    precision = (vox1 & vox2).sum() / vox1.sum()
    recall = (vox1 & vox2).sum() / vox2.sum()
    return iou, precision, recall


def evaluate_meshes(mesh1_path, mesh2_path, n_cd_samples=2000, vox_res=64):
    v1, f1 = pcu.load_mesh_vf(mesh1_path)
    v2, f2 = pcu.load_mesh_vf(mesh2_path)
    v1 = scale_to_unit_cube(v1)
    v2 = scale_to_unit_cube(v2)

    points1 = sample_surf_points(v1, f1, n_cd_samples).astype(float)
    points2 = sample_surf_points(v2, f2, n_cd_samples).astype(float)

    # pcu.save_mesh_v( "points1.obj", points1)
    # pcu.save_mesh_v( "points2.obj", points2)

    # chamfer distance
    cd = pcu.chamfer_distance(points1, points2)

    # voxelize
    # vox1 = voxelize(trimesh.Trimesh(v1, f1), 1. / vox_res).fill()
    # vox2 = voxelize(trimesh.Trimesh(v2, f2), 1. / vox_res).fill()
    vox1 = local_voxelize(trimesh.Trimesh(v1, f1), (0, 0, 0), 1. / vox_res, radius=vox_res // 2, fill=True)
    vox2 = local_voxelize(trimesh.Trimesh(v2, f2), (0, 0, 0), 1. / vox_res, radius=vox_res // 2, fill=True)
    vox1 = vox1.encoding._data
    vox2 = vox2.encoding._data

    iou, iou_precision, iou_recall = compute_iou(vox1, vox2)

    return {"CD": cd, "IOU": iou, "IOU_precision": iou_precision, "IOU_recall": iou_recall}


def evaluate_meshes_safe(mesh1_path, mesh2_path, n_cd_samples=2000, vox_res=64):
    v1, f1 = pcu.load_mesh_vf(mesh1_path)
    v2, f2 = pcu.load_mesh_vf(mesh2_path)
    v1 = scale_to_unit_cube(v1)
    v2 = scale_to_unit_cube(v2)

    points1 = sample_surf_points(v1, f1, n_cd_samples).astype(float)
    points2 = sample_surf_points(v2, f2, n_cd_samples).astype(float)

    vox2 = local_voxelize(trimesh.Trimesh(v2, f2), (0, 0, 0), 1. / vox_res, radius=vox_res // 2, fill=True)
    vox2 = vox2.encoding._data

    ret = {"CD": 100, "IOU": 0.0, "IOU_precision": 0.0, "IOU_recall": 0.0}

    for swap_idx in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
        for xyz_flip in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]:
            points1_trans = points1[:, swap_idx] * np.array(xyz_flip).reshape(1, 3)
            points2 = points2[:, [0, 1, 2]] # gt points stays the same
            cd = pcu.chamfer_distance(points1_trans, points2)
            if cd > ret["CD"]:
                continue

            # voxelize
            v1_trans = v1[:, swap_idx] * np.array(xyz_flip).reshape(1, 3)
            vox1_trans = local_voxelize(trimesh.Trimesh(v1_trans, f1), (0, 0, 0), 1. / vox_res, radius=vox_res // 2, fill=True)
            vox1_trans = vox1_trans.encoding._data
            iou, iou_precision, iou_recall = compute_iou(vox1_trans, vox2)

            if cd < ret["CD"]:
                ret["CD"] = cd
                ret["IOU"] = iou
                ret["IOU_precision"] = iou_precision
                ret["IOU_recall"] = iou_recall

    return ret


if __name__ == "__main__":
    mesh1_path = "/home/rliu/cvfiler04/rundi/sjc/experiments/rtmv/scene-00000-index-0_scale-100.0_train-view-True_train-depth-False_train-normal-False_augmentation-False_view-weight-10000_emp-wt-5000_emp-mtplr-20.0_emp_scl-10/source_data/scene.ply"
    mesh2_path = "/home/rliu/cvfiler04/rundi/sjc/experiments/rtmv/scene-00000-index-0_scale-100.0_train-view-True_train-depth-False_train-normal-False_augmentation-False_view-weight-10000_emp-wt-5000_emp-mtplr-20.0_emp_scl-10/data-nerf_synthetic-chair-train-r_2.png_scale-100.0_test_2000/mesh/step_100.obj"

    res = evaluate_meshes_safe(mesh1_path, mesh2_path)
    print(res)
