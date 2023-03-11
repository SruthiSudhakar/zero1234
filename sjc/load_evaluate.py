import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator
import json
import kornia
import cv2
import os
import math

from torchvision import transforms

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats, SurfaceNet
)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter, karras_t_schedule
from run_img_sampling import GDDPM, SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig, camera_pose, sample_near_eye

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)

from voxnerf.vis import stitch_vis, bad_vis as nerf_vis, vis_img
from voxnerf.data import load_googlescan_data, load_rtmv_data
from my3d import get_T, depth_smooth_loss

from mesh_metrics import evaluate_meshes

# # diet nerf
# import sys
# sys.path.append('./DietNeRF/dietnerf')
# from DietNeRF.dietnerf.run_nerf import get_embed_fn
# import torch.nn.functional as F

device_glb = torch.device("cuda")


def load_im(im_path):
    from PIL import Image
    import requests
    from torchvision import transforms
    if im_path.startswith("http"):
        response = requests.get(im_path)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content))
    else:
        im = Image.open(im_path).convert("RGB")
    tforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    inp = tforms(im).unsqueeze(0)
    return inp*2-1

def tsr_stats(tsr):
    return {
        "mean": tsr.mean().item(),
        "std": tsr.std().item(),
        "max": tsr.max().item(),
    }

class SJC(BaseConf):
    family:     str = "sd"
    gddpm:      GDDPM = GDDPM()
    sd:         SD = SD(
        variant="objaverse",
        prompt="A high quality photo of a delicious burger",
        im_path="data/nerf_synthetic/chair/train/r_2.png",
        scale=100.0
    )
    lr:         float = 0.05
    n_steps:    int = 10000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=False, bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=32, FoV=49.1, R=2.0)

    emptiness_scale:    int = 10
    emptiness_weight:   int = 1e4
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    grad_accum: int = 1

    depth_smooth_weight: float = 1e5

    depth_weight:       int = 0

    var_red:     bool = True

    train_view:         bool = True
    scene:              str = 'none'
    index:              int = 2

    view_weight:        int = 10000
    train_depth:        bool = False
    train_normal:       bool = False
    augmentation:       bool = False
    prefix:             str = 'exp'
    nerf_path:          str = "/proj/vondrick4/ruoshi/github/sjc/data/nerf_synthetic"

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()

        # family = cfgs.pop("family")
        # model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()
        vox = vox.to(device_glb)

        DATA_ROOT = "/home/rliu/cvfiler04/datasets/GoogleScannedObjects"
        phase = "test"
        path = os.path.join(DATA_ROOT, f"{phase}.json")

        with open(path) as f:   
            filenames = json.load(f)[:20]   

        res_dict = {}

        ckpt_root = '/home/rliu/cvfiler04/rundi/sjc/experiments/googlescan_{phase}_new/scene-{name}-index-0_scale-100.0_train-view-True_train-depth-False_train-normal-False_augmentation-False_view-weight-10000_emp-wt-5000_emp-mtplr-20.0_emp_scl-10'

        for name in filenames:
            path = os.path.join(ckpt_root, "ckpt", "step_10000.pt")
            mesh_path = path.replace("ckpt/step_10000.pt", "source_data/render/model_norm.obj")
            if not os.path.exists(path):
                continue

            state_dict = torch.load(path)
            vox.load_state_dict(state_dict)

            save_path = path[:-3] + f"_eval.obj"
            # try:
            vox.export_mesh(save_path, threshold_mult=8, kernel_size=7, kernel_type="avg", erosion_kernel_size=5)
            # except Exception as e:
            #     print(e)
            #     continue
            res = evaluate_meshes(mesh_path, save_path)
            print(f"name: {name}, res: {res}")
            save_path = save_path.replace(".obj", ".json")
            with open(save_path, "w") as f:
                json.dump(res, f)

            for k, v in res.items():
                if k not in res_dict:
                    res_dict[k] = []
                res_dict[k].append(v)

        for k, v in res_dict.items():
            print(f"{k}: {np.mean(v)}")


if __name__ == "__main__":
    seed_everything(0)
    dispatch(SJC)
    # evaluate_ckpt()
