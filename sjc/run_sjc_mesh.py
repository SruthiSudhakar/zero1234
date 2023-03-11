import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from imageio import imwrite
from pydantic import validator
import kornia
import cv2
from scipy.ndimage import gaussian_filter

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

from pytorch3d.utils import ico_sphere
import numpy as np
# from tqdm.notebook import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj
from voxnerf.data import load_blender
from voxnerf.utils import every

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

from voxnerf.vis import stitch_vis, bad_vis as nerf_vis, vis_img

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
    pose:       PoseConfig = PoseConfig(rend_hw=512, FoV=60.0, R=1.5, up='z')

    depth_weight:       int = 0

    var_red:     bool = True

    train_view:         bool = True
    scene:              str = 'chair'
    index:              int = 2

    emptiness_scale:    int = 10
    emptiness_weight:   int = 1e4
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    view_weight:        int = 10000
    train_depth:        bool = False
    train_normal:       bool = False
    augmentation:       bool = False
    prefix:             str = 'exp'
    nerf_path:          str = "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_synthetic"

    def run(self):
        cfgs = self.dict()

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("pose")
        poser = self.pose.make()

        sjc_3d(**cfgs, poser=poser, model=model)

def sjc_3d(poser, model: ScoreAdapter,
    lr, n_steps, depth_weight, var_red, train_view, scene, index, view_weight, train_depth, train_normal, 
    augmentation, prefix, nerf_path, **kwargs):

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1

    H, W = poser.H, poser.W
    Ks, poses, prompt_prefixes = poser.sample_train(n_steps)

    ts = model.us[30:-10]
    fuse = EarlyLoopBreak(5)

    same_noise = torch.randn(1, 4, H, W, device=model.device).repeat(bs, 1, 1, 1)

    normal_filter = SurfaceNet()
    normal_filter = normal_filter.to(model.device)

    folder_name = prefix + '/scene-%s-index-%d_scale-%s_train-view-%s_train-depth-%s_train-normal-%s_augmentation-%s_view-weight-%s' % \
                            (scene, index, model.scale, train_view, train_depth, train_normal, augmentation, view_weight)

    # if input is a view in nerf, load the view
    # if train_view:

    # load nerf view
    images_, _, poses_, mask_, fov_x = load_blender('test', scene=scene, path=nerf_path)
    # K_ = poser.get_K(H, W, fov_x * 180. / math.pi)
    K_ = poser.K
    input_image, input_K, input_pose, input_mask = images_[index], K_, poses_[index], mask_[index]
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

    ############## initialize mesh and optimizer ##############
    src_mesh = ico_sphere(4, model.device) # initialize a mesh

    # SoftRas
    sigma = 1e-4

    # assert H == W, 'Mesh renderer requires square image'
    raster_settings_soft = RasterizationSettings(
        image_size=512, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50, 
        perspective_correct=False, 
    )

    # Differentiable soft renderer using per vertex RGB colors for texture
    R_input, T_input = input_pose[:3, :3], input_pose[:3, -1]
    camera = FoVPerspectiveCameras(device=model.device, R=R_input[None, 1, ...], T=T_input[None, 1, ...])
    lights = PointLights(device=model.device, location=[[0.0, 0.0, -3.0]])
    renderer_textured = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(device=model.device, 
            cameras=camera,
            lights=lights)
    )

    # Optimize using rendered RGB image loss, rendered silhouette image loss, mesh 
    # edge loss, mesh normal consistency, and mesh laplacian smoothing
    losses = {"rgb": {"weight": 1.0, "values": []},
              "silhouette": {"weight": 1.0, "values": []},
              "edge": {"weight": 1.0, "values": []},
              "normal": {"weight": 0.01, "values": []},
              "laplacian": {"weight": 1.0, "values": []},
             }

    # We will learn to deform the source mesh by offsetting its vertices
    # The shape of the deform parameters is equal to the total number of vertices in 
    # src_mesh
    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=model.device, requires_grad=True)
    # deform_verts = torch.tensor(torch.randn(verts_shape, device=model.device) * 0.05, requires_grad=True)

    # We will also learn per vertex colors for our sphere mesh that define texture 
    # of the mesh
    sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=model.device, requires_grad=True)

    # The optimizer
    opt = torch.optim.AdamW([deform_verts, sphere_verts_rgb], lr=0.01, weight_decay=0.1)

    _, poses_light, _ = poser.sample_train(n_steps)

    print('==== initialized mesh for training ====')
    ###########################################################

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
            score_conds = model.img_emb(input_im)
            input_im_edge = kornia.filters.sobel(input_im)
            input_im_edge = input_im_edge - input_im_edge.mean()
            score_conds_edge = model.img_emb(input_im_edge)

        for i in range(n_steps):
            if fuse.on_break():
                break

            opt.zero_grad()

            # Deform the mesh
            new_src_mesh = src_mesh.offset_verts(deform_verts)
            
            # Add per vertex colors to texture the mesh
            new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb) 
            
            # Losses to smooth /regularize the mesh shape
            loss = {k: torch.tensor(0.0, device=model.device) for k in losses}
            update_mesh_shape_prior_losses(new_src_mesh, loss)

            if train_view:

                # supervise with input view
                # if i < 100 or i % 10 == 0:

                image_, depth_ = render_one_view(new_src_mesh, renderer_textured, input_K, input_pose, poses_light[i], model.device, 512)

                rgb_loss = ((image_[image_mask] - input_image[image_mask]) ** 2).mean()
                depth_loss = 0.
                input_loss = (rgb_loss + depth_loss * 10.) * float(view_weight)
                # input_loss = rgb_loss * 10.
                input_loss.backward(retain_graph=True)

                # save render view
                if i % 100 == 0:
                    with torch.no_grad():
                        print('rgb_loss: %.5f    depth_loss: %.5f    background_loss: %.5f' % (rgb_loss.item(), depth_loss.item(), background_loss.item()))
                        metric.put_artifact("input_view", ".png", lambda fn: imwrite(fn, denorm_img(image_)))


            # y: [1, 4, 64, 64] depth: [64, 64]  ws: [n, 4096]
            image, depth = render_one_view(new_src_mesh, renderer_textured, Ks[i], poses[i], poses_light[i], model.device, 512)

            with torch.enable_grad():
                y = model.encode(image)

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
            # print(deform_verts)

            metric.put_scalars(**tsr_stats(y))

            if i % 1000 == 0:
                with EventStorage(model.im_path.replace('/', '-') + '_scale-' + str(model.scale) + "_test"):
                    evaluate(model, poser, renderer_textured, src_mesh, deform_verts, sphere_verts_rgb, poses_light[i])

            if every(pbar, percent=1):
                with torch.no_grad():
                    if isinstance(model, StableDiffusion):
                        vis_routine(metric, image, depth_value)

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
        with EventStorage(folder_name + "_test"):
            evaluate(model, poser, renderer_textured, src_mesh, deform_verts, sphere_verts_rgb, poses_light[i])

        metric.step()

        hbeat.done()


@torch.no_grad()
def evaluate(score_model, poser, renderer, src_mesh, deform_verts, sphere_verts_rgb, poses_light):
    H, W = poser.H, poser.W
    K, poses = poser.sample_test(100)

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    
    # Add per vertex colors to texture the mesh
    new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb) 

    num_imgs = len(poses)

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break

        pose = poses[i]
        poses_light = np.eye(4)
        poses_light[:3, -1] = np.array([0, 0, 3.])
        image, depth = render_one_view(new_src_mesh, renderer, K, pose, poses_light, score_model.device, 512)
        vis_routine(metric, image, depth)

        metric.step()
        hbeat.beat()

    metric.flush_history()

    metric.put_artifact(
        "view_seq", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "view")[1])
    )

    metric.step()

def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)
    
    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)
    
    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")

def render_one_view(new_src_mesh, renderer, K, pose, pose_light, device, render_hw=512):

    # prepare light
    T_light = pose_light[:3, -1]
    lights = PointLights(device=device, location=T_light[None, :])

    pose = np.linalg.inv(pose)

    # prepare camera
    R_camera, T_camera = pose[:3, :3], pose[:3, -1]
    R_camera = np.linalg.inv(R_camera)
    K_temp = np.eye(4)
    K_temp[:3, :3] = K
    K = K_temp
    K[0, 0] /= render_hw
    K[0, 2] /= render_hw
    K[1, 1] /= render_hw
    K[1, 2] /= render_hw
    K[2, 2] = -K[2, 2]

    camera = FoVPerspectiveCameras(device=device, R=R_camera[None, :], T=T_camera[None, :])
    # print(R_camera, T_camera, K)

    images_predicted = renderer(new_src_mesh, cameras=camera, lights=lights)

    # normalize and reshape image
    images_predicted = images_predicted.permute(0, 3, 1, 2)[:, :3, :, :]
    images_predicted = images_predicted * 2 - 1.

    return images_predicted, torch.zeros(1) # depth not implemented yet

def scene_box_filter(ro, rd, aabb):
    _, t_min, t_max = ray_box_intersect(ro, rd, aabb)
    # do not render what's behind the ray origin
    t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
    return ro, rd, t_min, t_max


def vis_routine(metric, y, depth):
    # pane = nerf_vis(y, depth, final_H=256)
    img = denorm_img(y)
    # depth = depth.cpu().numpy()
    # metric.put_artifact("view", ".png", lambda fn: imwrite(fn, pane))
    metric.put_artifact("img", ".png", lambda fn: imwrite(fn, img))
    # metric.put_artifact("depth", ".npy", lambda fn: np.save(fn, depth))


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

def denorm_img(img):
    img = (img + 1.) / 2.
    img = (img.permute(0, 2, 3, 1)[0].detach().cpu().numpy() * 255.).astype(np.uint8)
    return img

def latest_ckpt():
    ts, ys = read_stats("./", "ckpt")
    assert len(ys) > 0
    return ys[-1]


if __name__ == "__main__":
    seed_everything(0)
    dispatch(SJC)
    # evaluate_ckpt()
