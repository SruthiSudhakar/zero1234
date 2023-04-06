import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator
import kornia
import cv2
import math
import pdb
import json 
import os
import plotly 
from torchvision import transforms
import wandb

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak,
    get_event_storage, get_heartbeat, read_stats
)
from my.config import BaseConf, dispatch, optional_load_config
from my.utils.seed import seed_everything

from adapt import ScoreAdapter, karras_t_schedule
from run_img_sampling import SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig, camera_pose, sample_near_eye

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis, vis_img
from voxnerf.data import load_blender
from my3d import get_T, depth_smooth_loss

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
    sd:         SD = SD(
        variant="objaverse",
        scale=100.0
    )
    lr:         float = 0.05
    n_steps:    int = 10000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=3,
        blend_bg_texture=False, bg_texture_hw=4,
        bbox_len=1.0, time_lr=0.0001
    )
    pose:       PoseConfig = PoseConfig(rend_hw=32, FoV=49.1, R=2.0)

    emptiness_scale:    int = 10
    emptiness_weight:   int = 0
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    grad_accum: int = 1

    depth_smooth_weight: float = 1e5
    near_view_weight: float = 1e5

    depth_weight:       int = 0

    var_red:     bool = True

    train_view:         bool = True
    scene:              str = 'chair'
    index:              int = 2

    view_weight:        int = 10000
    prefix:             str = 'exp'
    nerf_path:          str = "data/nerf_wild"

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

        cfgs.pop("pose")
        poser = self.pose.make()
        folder_name = self.prefix + '/%s_scale-%s_train-view-%s_view-weight-%s_depth-smooth-wt-%s_near-view-wt-%s_time-lr_%s' % \
                            (self.scene, model.scale, self.train_view, self.view_weight, self.depth_smooth_weight, self.near_view_weight, self.vox.time_lr)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open(f'{folder_name}/data.json', 'w') as fp:
            json.dump(cfgs, fp)
            
        run = wandb.init(
            project=f"{self.scene}_timelr_{self.vox.time_lr}",
            notes="dzero123",
        )
        wandb.config = cfgs

        sjc_3d(**cfgs, poser=poser, model=model, vox=vox, time_lr=vox.time_lr)

def sjc_3d(poser, vox, model: ScoreAdapter, time_lr,
    lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
    depth_weight, var_red, train_view, scene, view_weight, prefix, nerf_path, \
    depth_smooth_weight, near_view_weight, grad_accum, **kwargs):

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1
    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)
    opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

    H, W = poser.H, poser.W
    Ks, poses, prompt_prefixes = poser.sample_train(n_steps)

    ts = model.us[30:-10]
    fuse = EarlyLoopBreak(5)

    same_noise = torch.randn(1, 4, H, W, device=model.device).repeat(bs, 1, 1, 1)

    folder_name = prefix + '/%s_scale-%s_train-view-%s_view-weight-%s_depth-smooth-wt-%s_near-view-wt-%s_time-lr_%s' % \
                            (scene, model.scale, train_view, view_weight, depth_smooth_weight, near_view_weight, time_lr)
    
    # load nerf view
    images_, _, poses_, times_, mask_, fov_x = load_blender('train', scene=scene, path=nerf_path)
    print(f"NUMBER OF IMAGES: {len(images_)}")
    input_im = []
    input_im_transformed = []
    with torch.no_grad():
        tforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((256, 256))
        ])
        for img in images_:
            img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            img = torch.as_tensor(img, dtype=float, device=device_glb)
            img = img.permute(2, 0, 1)[None, :, :]
            img = img * 2. - 1.
            input_im.append(img)
            input_im_transformed.append(tforms(img))
    opt.zero_grad()

    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage(folder_name) as metric:
            
        for i in range(n_steps):
            if fuse.on_break():
                break

            if train_view:
                # for inner_index in range(len(images_)):
                inner_index = np.random.randint(0,len(images_))
                # get input embedding
                model.clip_emb = model.model.get_learned_conditioning(input_im_transformed[inner_index].float()).tile(1,1,1).detach()
                model.vae_emb = model.model.encode_first_stage(input_im_transformed[inner_index].float()).mode().detach()

                # K_ = poser.get_K(H, W, fov_x * 180. / math.pi)
                K_ = poser.K
                _, input_K, input_pose, input_mask, input_time = images_[inner_index], K_, poses_[inner_index], mask_[inner_index], times_[inner_index]
                input_pose[:3, -1] = input_pose[:3, -1] / np.linalg.norm(input_pose[:3, -1]) * poser.R
                background_mask, image_mask = input_mask == 0., input_mask != 0.
                image_mask = cv2.resize(image_mask.astype(np.float32), dsize=(256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
                background_mask = cv2.resize(background_mask.astype(np.float32), dsize=(H, W), interpolation=cv2.INTER_NEAREST).astype(bool)

                # to torch tensor
                input_time = torch.as_tensor(input_time, dtype=float, device=device_glb)
                image_mask = torch.as_tensor(image_mask, dtype=bool, device=device_glb)
                image_mask = image_mask[None, None, :, :].repeat(1, 3, 1, 1)
                background_mask = torch.as_tensor(background_mask, dtype=bool, device=device_glb)

                # print('==== loaded input view for training ====')

                # supervise with input view
                # if i < 100 or i % 10 == 0:
                with torch.enable_grad():
                    y_, depth_, ws_, displacements_ = render_one_view(vox, aabb, H, W, input_K, input_pose, input_time, return_w=True)
                    y_ = model.decode(y_)
                rgb_loss = ((y_ - input_im[inner_index]) ** 2).mean()

                #need to use the displacements for some type of loss as done in d-nerf

                # depth smoothness loss
                input_smooth_loss = depth_smooth_loss(depth_) * depth_smooth_weight * 0.1
                input_smooth_loss.backward(retain_graph=True)

                input_loss = rgb_loss * float(view_weight)
                input_loss.backward(retain_graph=True)
                if train_view and i % 100 == 0:
                    metric.put_artifact(f"input_view_{inner_index}", ".png", lambda fn: imwrite(fn, torch_samps_to_imgs(y_)[0]))

            with torch.no_grad():
                # get input embedding
                model.clip_emb = model.model.get_learned_conditioning(input_im_transformed[i%len(times_)].float()).tile(1,1,1).detach()
                model.vae_emb = model.model.encode_first_stage(input_im_transformed[i%len(times_)].float()).mode().detach()

            # y: [1, 4, 64, 64] depth: [64, 64]  ws: [n, 4096]
            y, depth, ws, displacements = render_one_view(vox, aabb, H, W, Ks[i], poses[i], times_[i%len(times_)], return_w=True)

            # near-by view
            eye = poses[i][:3, -1]
            near_eye = sample_near_eye(eye)
            near_pose = camera_pose(near_eye, -near_eye, poser.up)
            y_near, depth_near, ws_near, displacements_near = render_one_view(vox, aabb, H, W, Ks[i], near_pose, times_[i%len(times_)], return_w=True)
            near_loss = ((y_near - y).abs().mean() + (depth_near - depth).abs().mean()) * near_view_weight
            near_loss.backward(retain_graph=True)

            # get T from input view
            pose = poses[i]
            T_target = pose[:3, -1]
            T_cond = input_pose[:3, -1]
            T = get_T(T_target, T_cond).to(model.device)

            if isinstance(model, StableDiffusion):
                pass
            else:
                y = torch.nn.functional.interpolate(y, (target_H, target_W), mode='bilinear')

            with torch.no_grad():
                chosen_σs = np.random.choice(ts, bs, replace=False)
                chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
                chosen_σs = torch.as_tensor(chosen_σs, device=model.device, dtype=torch.float32)
                # chosen_σs = us[i]

                noise = torch.randn(bs, *y.shape[1:], device=model.device)

                zs = y + chosen_σs * noise

                score_conds = model.img_emb(input_im_transformed[i%len(times_)], conditioning_key='hybrid', T=T)

                Ds = model.denoise_objaverse(zs, chosen_σs, score_conds)

                if var_red:
                    grad = (Ds - y) / chosen_σs
                else:
                    grad = (Ds - zs) / chosen_σs

                grad = grad.mean(0, keepdim=True)

            y.backward(-grad, retain_graph=True)

            emptiness_loss = (torch.log(1 + emptiness_scale * ws) * (-1 / 2 * ws)).mean() # negative emptiness loss
            emptiness_loss = emptiness_weight * emptiness_loss
            # if emptiness_step * n_steps <= i:
            #     emptiness_loss *= emptiness_multiplier
            emptiness_loss = emptiness_loss * (1. + emptiness_multiplier * i / n_steps)
            # emptiness_loss.backward(retain_graph=True)

            # depth smoothness loss
            smooth_loss = depth_smooth_loss(depth) * depth_smooth_weight

            if i >= emptiness_step * n_steps:
                smooth_loss.backward(retain_graph=True)

            depth_value = depth.clone()

            if i % grad_accum == (grad_accum-1):
                opt.step()
                # print(f"INNER INDEX: {inner_index}")
                if vox._time[0].weight == None:
                    print("weight: None")
                else:
                    print("weight: ", sum(vox._time[0].weight))
                if vox._time[0].weight.grad == None:
                    print("grad: None")      
                else:
                    print('grad: ',sum(vox._time[0].weight.grad))       
                # pdb.set_trace()
                opt.zero_grad()

            metric.put_scalars(**tsr_stats(y))

            if i % 1000 == 0 and i != 0:
                with EventStorage(model.im_path.replace('/', '-') + '_scale-' + str(model.scale) + "_test"):
                    evaluate(model, vox, poser, times_, input_im_transformed, inner_index)
                metric.put_artifact(
                    "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
                )
            if every(pbar, percent=1):
                with torch.no_grad():
                    if isinstance(model, StableDiffusion):
                        y = model.decode(y)
                    vis_routine(metric, y, depth_value, inner_index)

            metric.step()
            pbar.update()
            pbar.set_description(f"frame:{inner_index} rgb loss: {str(rgb_loss.item())[:7]}  input_smooth_loss: {str(input_smooth_loss.item())[:7]} input_loss: {str(input_loss.item())[:7]}  near_loss: {str(near_loss.item())[:7]}  emptiness_loss: {str(emptiness_loss.item())[:7]}  smooth_loss: {str(smooth_loss.item())[:7]} ")
            wandb.log({"frame": inner_index, "rgb loss": rgb_loss.item(), "input_smooth_loss": input_smooth_loss.item()})
            hbeat.beat()

        metric.put_artifact(
            "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
        )
        with EventStorage("test"):
            evaluate(model, vox, poser, times_, input_im_transformed, inner_index)

        metric.step()

        hbeat.done()


@torch.no_grad()
def evaluate(score_model, vox, poser, times_, input_im_transformed, inner_index=0):
    H, W = poser.H, poser.W
    vox.eval()
    # pdb.set_trace()
    K, poses = poser.sample_test(100)

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    num_imgs = len(poses)

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break
        
        score_model.clip_emb = score_model.model.get_learned_conditioning(input_im_transformed[inner_index].float()).tile(1,1,1).detach()
        score_model.vae_emb = score_model.model.encode_first_stage(input_im_transformed[inner_index].float()).mode().detach()


        pose = poses[i]
        # print("IN THE EVALUATE STEP")
        
        time = times_[inner_index]
        y, depth, disp = render_one_view(vox, aabb, H, W, K, pose, time)

        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        vis_routine(metric, y, depth, inner_index)

        metric.step()
        hbeat.beat()

        metric.flush_history()

        metric.put_artifact(
            "view_seq", ".mp4",
            lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "view")[1])
        )

        metric.step()


def render_one_view(vox, aabb, H, W, K, pose, frame_time, return_w=False):
    # print(f"IN RENDER_ONE_VIEW: {frame_time}")
    # pdb.set_trace()
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)

    #adding time stuff here
    frame_time = frame_time * torch.ones_like(rd[...,:1])

    # rgbs, depth, weights, displacements, og_pts, og_pds, og_tp = render_ray_bundle(vox, ro, rd, t_min, t_max, frame_time)
    rgbs, depth, weights, displacements = render_ray_bundle(vox, ro, rd, t_min, t_max, frame_time)
    
    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    weights = rearrange(weights, "N (h w) 1 -> N h w", h=H, w=W)
    if return_w:
        return rgbs, depth, weights, displacements#, og_pts, og_pds, og_tp
    else:
        return rgbs, depth, displacements#, og_pts, og_pds, og_tp


def scene_box_filter(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth, inner_index):
    # y = torch.nn.functional.interpolate(y, 512, mode='bilinear', antialias=True)
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]
    depth = depth.cpu().numpy()
    metric.put_artifact(f"{inner_index}/view", ".png", lambda fn: imwrite(fn, pane))
    metric.put_artifact(f"{inner_index}/img", ".png", lambda fn: imwrite(fn, im))
    metric.put_artifact(f"{inner_index}/depth", ".npy", lambda fn: np.save(fn, depth))


def evaluate_ckpt():
    cfg = optional_load_config(fname="full_config_objaverse.yml")
    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()

    pbar = tqdm(range(1))

    with EventStorage(model.prompt.replace(' ', '-')), HeartBeat(pbar):
        ckpt_fname = latest_ckpt()
        state = torch.load(ckpt_fname, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)
        pdb.set_trace()
        with EventStorage(model.prompt.replace(' ', '-') + "_test"):
            evaluate(model, vox, poser)


def latest_ckpt():
    ts, ys = read_stats("./", "ckpt")
    assert len(ys) > 0
    return ys[-1]


if __name__ == "__main__":
    seed_everything(0)
    dispatch(SJC)
    # evaluate_ckpt()
