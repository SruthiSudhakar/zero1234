import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator
import kornia
import cv2, os
import json
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
from pose import PoseConfig
from mesh_metrics import evaluate_meshes

from run_nerf import VoxConfig
from voxnerf.utils import every
from voxnerf.render import (
    as_torch_tsrs, rays_from_img, ray_box_intersect, render_ray_bundle
)
from voxnerf.vis import stitch_vis, bad_vis as nerf_vis, vis_img
from voxnerf.data import load_blender
from voxnerf.data import load_googlescan_data, load_wild

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
        blend_bg_texture=True, bg_texture_hw=4,
        bbox_len=1.0
    )
    pose:       PoseConfig = PoseConfig(rend_hw=64, FoV=49.1, R=2.0)

    emptiness_scale:    int = 10
    emptiness_weight:   int = 1e4
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    depth_weight:       int = 0

    var_red:     bool = True

    train_view:         bool = True
    scene:              str = 'chair'
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

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        sjc_3d(**cfgs, poser=poser, model=model, vox=vox)

def sjc_3d(poser, vox, model: ScoreAdapter,
    lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step, emptiness_multiplier,
    depth_weight, var_red, train_view, scene, index, view_weight, train_depth, train_normal, 
    augmentation, prefix, nerf_path, **kwargs):

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

    normal_filter = SurfaceNet()
    normal_filter = normal_filter.to(model.device)

    folder_name = prefix + '/scene-%s-index-%d_scale-%s_train-view-%s_train-depth-%s_train-normal-%s_augmentation-%s_view-weight-%s' % \
                            (scene, index, model.scale, train_view, train_depth, train_normal, augmentation, view_weight)

    os.makedirs(folder_name, exist_ok=True)
    input_image, input_pose, input_mask, _ = load_wild(nerf_path, scene, index)
    try:
        os.symlink(os.path.join(nerf_path, scene), os.path.join(folder_name, 'source_data'))
    except Exception as e:
        pass

    K_ = poser.K
    input_K = K_
    # input_image, input_K, input_pose, input_mask = images_[index], K_, poses_[index], mask_[index]
    input_pose[:3, -1] = input_pose[:3, -1] / np.linalg.norm(input_pose[:3, -1]) * 2.0
    background_mask, image_mask = input_mask == 0., input_mask != 0.
    input_image = cv2.resize(input_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    image_mask = cv2.resize(image_mask.astype(np.float32), dsize=(512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
    background_mask = cv2.resize(background_mask.astype(np.float32), dsize=(H, W), interpolation=cv2.INTER_NEAREST).astype(bool)

    # to torch tensor
    input_image = torch.as_tensor(input_image, dtype=float, device=device_glb)
    input_image = input_image.permute(2, 0, 1)[None, :, :]
    input_image = input_image * 2. - 1.
    image_mask = torch.as_tensor(image_mask, dtype=bool, device=device_glb)
    image_mask = image_mask[None, None, :, :].repeat(1, 3, 1, 1)
    background_mask = torch.as_tensor(background_mask, dtype=bool, device=device_glb)

    print('==== loaded input view for training ====')

    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage(folder_name) as metric:
        
        with torch.no_grad():
            # if train_view:
            if augmentation:
                tforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop((224, 224)),
                    transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomPerspective(distortion_scale=0.6, p=0.5)
                ])
            else:
                tforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop((224, 224))
                ])
            input_im = tforms(input_image)
            # else:
            #     input_im = load_im(model.im_path).to(model.device)
            score_conds = model.img_emb(input_im, conditioning_key="c_crossattan")
            input_im_edge = kornia.filters.sobel(input_im)
            input_im_edge = input_im_edge - input_im_edge.mean()
            score_conds_edge = model.img_emb(input_im_edge, conditioning_key="c_crossattan")

        for i in range(n_steps):
            if fuse.on_break():
                break

            opt.zero_grad()

            if train_view:

                # supervise with input view
                # if i < 100 or i % 10 == 0:
                y_, depth_, ws_ = render_one_view(vox, aabb, H, W, input_K, input_pose, return_w=True)
                with torch.enable_grad():
                    y_ = model.decode(y_)
                rgb_loss = ((y_[image_mask] - input_image[image_mask]) ** 2).mean()
                background_density = ws_[background_mask[None, :, :].repeat(ws_.shape[0], 1, 1)]
                # background_loss = torch.log(1. + background_density).mean()
                background_density = background_density.view([ws_.shape[0], -1])
                background_loss = background_density.sum(0).mean()

                # depth_loss = -depth_[background_mask].mean()
                depth_loss = -torch.log(1. + depth_[background_mask]).mean()

                input_loss = (rgb_loss + depth_loss * 10.) * float(view_weight)
                # input_loss = rgb_loss * 10.
                input_loss.backward(retain_graph=True)

                # save render view
                if i % 100 == 0:
                    with torch.no_grad():
                        print('rgb_loss: %.5f    depth_loss: %.5f    background_loss: %.5f' % (rgb_loss.item(), depth_loss.item(), background_loss.item()))
                        metric.put_artifact("input_view", ".png", lambda fn: imwrite(fn, torch_samps_to_imgs(y_)[0]))


            # y: [1, 4, 64, 64] depth: [64, 64]  ws: [n, 4096]
            y, depth, ws = render_one_view(vox, aabb, H, W, Ks[i], poses[i], return_w=True)

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
                Ds = model.denoise(zs, chosen_σs, **score_conds)

                if var_red:
                    grad = (Ds - y) / chosen_σs
                else:
                    grad = (Ds - zs) / chosen_σs

                grad = grad.mean(0, keepdim=True)

            y.backward(-grad, retain_graph=True)

            if depth_weight > 0:
                center_depth = depth[7:-7, 7:-7]
                border_depth_mean = (depth.sum() - center_depth.sum()) / (64*64-50*50)
                center_depth_mean = center_depth.mean()
                depth_diff = center_depth_mean - border_depth_mean
                depth_loss = - torch.log(depth_diff + 1e-12)
                depth_loss = depth_weight * depth_loss
                depth_loss.backward(retain_graph=True)

            emptiness_loss = torch.log(1 + emptiness_scale * ws).mean()
            emptiness_loss = emptiness_weight * emptiness_loss
            if emptiness_step * n_steps <= i:
                emptiness_loss *= emptiness_multiplier
            emptiness_loss.backward(retain_graph=True)

            depth_value = depth.clone()

            if train_depth:
                # depth texture loss
                depth = depth[None, None, :, :].repeat([1, 4, 1, 1])
                depth = (depth - depth.mean()) / depth.std() # normalized depth map
                # if i % 20 == 0:
                #     print("[depth stats] min: %.3f, max: %.3f, mean: %.3f, std: %.3f" % \
                #         (depth.min().item(), depth.max().item(), depth.mean().item(), depth.std().item()))

                if isinstance(model, StableDiffusion):
                    pass
                else:
                    depth = torch.nn.functional.interpolate(depth, (target_H, target_W), mode='bilinear')

                with torch.no_grad():
                    chosen_σs_depth = np.random.choice(ts, bs, replace=False)
                    chosen_σs_depth = chosen_σs_depth.reshape(-1, 1, 1, 1)
                    chosen_σs_depth = torch.as_tensor(chosen_σs_depth, device=model.device, dtype=torch.float32)
                    # chosen_σs = us[i]

                    noise_depth = torch.randn(bs, *depth.shape[1:], device=model.device)

                    zs_depth = depth + chosen_σs_depth * noise_depth
                    Ds_depth = model.denoise(zs_depth, chosen_σs_depth, **score_conds_edge)

                    if var_red:
                        grad_depth = (Ds_depth - depth) / chosen_σs_depth
                    else:
                        grad_depth = (Ds_depth - zs_depth) / chosen_σs_depth

                    grad_depth = grad_depth.mean(0, keepdim=True)

                depth.backward(-grad_depth, retain_graph=True)

            if train_normal:
                # surface normal loss
                normal = normal_filter(torch.nn.functional.interpolate(depth_value[None, None, :, :],\
                    (target_H + 2, target_W + 2), mode='bilinear'))[0].mean(dim=1)\
                    [:, None, :, :].repeat(1, 4, 1, 1).float()
                normal = normal - normal.mean() # normalize
                # if i % 20 == 0:
                #     print("[normal stats] min: %.3f, max: %.3f, mean: %.3f, std: %.3f" % \
                #         (normal.min().item(), normal.max().item(), normal.mean().item(), normal.std().item()))

                with torch.no_grad():
                    chosen_σs_normal = np.random.choice(ts, bs, replace=False)
                    chosen_σs_normal = chosen_σs_normal.reshape(-1, 1, 1, 1)
                    chosen_σs_normal = torch.as_tensor(chosen_σs_normal, device=model.device, dtype=torch.float32)
                    # chosen_σs = us[i]

                    noise_normal = torch.randn(bs, *normal.shape[1:], device=model.device)

                    zs_normal = normal + chosen_σs_normal * noise_normal
                    Ds_normal = model.denoise(zs_normal, chosen_σs_normal, **score_conds_edge)

                    if var_red:
                        grad_normal = (Ds_normal - normal) / chosen_σs_normal
                    else:
                        grad_normal = (Ds_normal - zs_normal) / chosen_σs_normal

                    grad_normal = grad_normal.mean(0, keepdim=True)

                normal.backward(-grad_normal, retain_graph=True)

            opt.step()

            metric.put_scalars(**tsr_stats(y))

            if i % 2000 == 0 and i != 0:
                with EventStorage(model.im_path.replace('/', '-') + '_scale-' + str(model.scale) + "_test"):
                    evaluate(model, vox, poser)

            if every(pbar, percent=1):
                with torch.no_grad():
                    if isinstance(model, StableDiffusion):
                        y = model.decode(y)
                    vis_routine(metric, y, depth_value)

            # if every(pbar, step=2500):
            #     metric.put_artifact(
            #         "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
            #     )
            #     with EventStorage("test"):
            #         evaluate(model, vox, poser)

            metric.step()
            pbar.update()
            pbar.set_description(model.im_path)
            hbeat.beat()

        metric.put_artifact(
            "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
        )
        with EventStorage("test"):
            evaluate(model, vox, poser)

        metric.step()

        hbeat.done()


@torch.no_grad()
def evaluate(score_model, vox, poser):
    H, W = poser.H, poser.W
    vox.eval()
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

        pose = poses[i]
        y, depth = render_one_view(vox, aabb, H, W, K, pose)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        vis_routine(metric, y, depth)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "view_seq", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "view")[1])
    )

    # save mesh
    save_mesh_path = metric.put_artifact(
        "mesh", ".obj",
        lambda fn: vox.export_mesh(fn)
    )

    # # evaluate CD and IoU
    # res = evaluate_meshes(score_model.input_mesh_path, save_mesh_path)

    # def save_json(fn, data):
    #     with open(fn, "w") as f:
    #         json.dump(data, f)

    # metric.put_artifact(
    #     "mesh_metrics", ".json",
    #     lambda fn: save_json(fn, res)
    # )

    metric.step()


def render_one_view(vox, aabb, H, W, K, pose, return_w=False):
    N = H * W
    ro, rd = rays_from_img(H, W, K, pose)
    ro, rd, t_min, t_max = scene_box_filter(ro, rd, aabb)
    assert len(ro) == N, "for now all pixels must be in"
    ro, rd, t_min, t_max = as_torch_tsrs(vox.device, ro, rd, t_min, t_max)
    rgbs, depth, weights = render_ray_bundle(vox, ro, rd, t_min, t_max)

    rgbs = rearrange(rgbs, "(h w) c -> 1 c h w", h=H, w=W)
    depth = rearrange(depth, "(h w) 1 -> h w", h=H, w=W)
    weights = rearrange(weights, "N (h w) 1 -> N h w", h=H, w=W)
    if return_w:
        return rgbs, depth, weights
    else:
        return rgbs, depth


def scene_box_filter(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth):
    pane = nerf_vis(y, depth, final_H=256)
    im = torch_samps_to_imgs(y)[0]
    depth = depth.cpu().numpy()
    metric.put_artifact("view", ".png", lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png", lambda fn: imwrite(fn, im))
    metric.put_artifact("depth", ".npy", lambda fn: np.save(fn, depth))


def evaluate_ckpt():
    cfg = optional_load_config(fname="full_config_3d.yml")
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
