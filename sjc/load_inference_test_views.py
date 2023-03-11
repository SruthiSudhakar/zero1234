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
import imageio

from run_sjc_objaverse import render_one_view

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
        variant="image",
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
    pose:       PoseConfig = PoseConfig(rend_hw=64, FoV=49.1, R=2.0)
    data_root:          str = "/home/rliu/Desktop/cvfiler04/datasets/GoogleScannedObjects"
    res_path:           str = "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc_rundi/experiments/googlescan_test_new_baseline"

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()
        vox = vox.to(device_glb)

        cfgs.pop("pose")
        poser = self.pose.make()

        if 'GoogleScannedObjects' in cfgs['data_root']:
            phase = "test"
            path = os.path.join(cfgs['data_root'], f"{phase}.json")

            with open(path) as f:   
                filenames = json.load(f)[:20]   
        else:
            filenames = os.listdir(cfgs['data_root'])

        print(filenames)

        for i_file, name in tqdm(enumerate(filenames), total=len(filenames)):

            res_path = cfgs['res_path']
            
            if 'GoogleScannedObjects' in cfgs['data_root']:
                num_test_view = 10
            else:
                num_test_view = 17

            ckpt_root = os.path.join(res_path, 'scene-%s-index-0_scale-100.0_train-view-True_train-depth-False_train-normal-False_augmentation-False_view-weight-10000' % name)

            path = os.path.join(ckpt_root, "ckpt", "step_10000.pt")
            test_view_path = os.path.join(ckpt_root, 'test_views')
            os.makedirs(test_view_path, exist_ok=True)

            if not os.path.exists(path):
                continue

            state_dict = torch.load(path)
            vox.load_state_dict(state_dict)

            for i_view in range(num_test_view):

                if 'GoogleScannedObjects' in cfgs['data_root']:
                    image, pose, _, _ = load_googlescan_data(cfgs['data_root'], name, i_view, split='render_mvs_25')
                else:
                    image, pose, _, _ = load_rtmv_data(cfgs['data_root'], name, i_view)
                    # print('here')
                y, depth, ws = render_one_view(vox, vox.aabb.T.cpu().numpy(), poser.H, poser.W, poser.K,\
                                               pose, return_w=True)
                y = model.decode(y)
                pane = nerf_vis(y.detach(), depth.detach(), final_H=256)
                im = torch_samps_to_imgs(y)[0]
                imageio.imwrite(os.path.join(test_view_path, '%d.png' % i_view), im)


if __name__ == "__main__":
    seed_everything(0)
    dispatch(SJC)
    # evaluate_ckpt()
