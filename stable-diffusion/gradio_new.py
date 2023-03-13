'''
conda activate zero123
'''

import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import torch
from contextlib import nullcontext
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from torch import autocast
from torchvision import transforms


_SHOW_INTERMEDIATE = False
_GPU_INDEX = 0
# _GPU_INDEX = 2

_TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'

_DESCRIPTION = '''
This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
Check out our project webpage and paper if you want to learn more about the method!
'''

_ARTICLE = '''
## Model Card
TBD
## Limitations
TBD
'''


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location=device)
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def preprocess_image(input_im, preprocess):

    print('old input_im:', input_im.size)

    if preprocess:
        input_im = load_and_preprocess(input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].

    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: very important, thresholding background
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print('new input_im:', lo(input_im))
    return input_im


# def main_vis(cam_vis, raw_im, preprocess=True):

#     input_im = preprocess_image(raw_im, preprocess)

#     show_in_im1 = (input_im * 255).astype(np.uint8)
#     show_in_im2 = Image.fromarray(show_in_im1)
#     new_fig = cam_vis.encode_image(show_in_im1)

#     return (new_fig, show_in_im2)


def main_run(model, device, cam_vis, vis_only, raw_im=None,
             preprocess=True, x=0.0, y=0.0, z=0.0,
             scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256):

    input_im = preprocess_image(raw_im, preprocess)

    show_in_im1 = (input_im * 255.0).astype(np.uint8)
    show_in_im2 = Image.fromarray(show_in_im1)

    cam_vis.polar_change(x)
    cam_vis.azimuth_change(y)
    cam_vis.radius_change(z)
    cam_vis.encode_image(show_in_im1)
    new_fig = cam_vis.update_figure()

    if vis_only:
        return (new_fig, show_in_im2)

    else:
        # yield new_fig

        input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
        input_im = input_im * 2 - 1
        input_im = transforms.functional.resize(input_im, [h, w])

        sampler = DDIMSampler(model)
        used_x = -x  # NOTE: polar makes more sense in Basile's opinion this way!
        x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w,
                                      ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)

        output_ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

        return (new_fig, output_ims, show_in_im2)
        # yield output_ims
        # yield show_in_im2


def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])
    print('camera_R:', lo(camera_R).v)

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T


class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def polar_change(self, value):
        self._polar = value
        # return self.update_figure()

    def azimuth_change(self, value):
        self._azimuth = value
        # return self.update_figure()

    def radius_change(self, value):
        self._radius = value
        # return self.update_figure()

    def encode_image(self, raw_image):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            print('x:', lo(x))
            print('y:', lo(y))
            print('z:', lo(z))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=self._8bit_image,
                cmin=0,
                cmax=255,
                colorscale=self._image_colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3))

            base_radius = 3.0
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(
                0.0, 0.0, base_radius, fov_deg)  # (5, 3).
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius, fov_deg)  # (5, 3).
            print('input_cone:', lo(input_cone).v)
            print('output_cone:', lo(output_cone).v)

            for (cone, clr) in [(input_cone, 'green'), (output_cone, 'blue')]:
                for edge in edges:
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(color=clr, width=3)))

            # look at center of scene
            fig.update_layout(
                # width=640,
                # height=480,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0)),
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis=dict(
                        range=[-base_radius - 0.5, base_radius + 0.5],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    yaxis=dict(
                        range=[-base_radius - 0.5, base_radius + 0.5],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    zaxis=dict(
                        range=[-base_radius - 0.5, base_radius + 0.5],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks='')))

        self._fig = fig
        return fig


def run_demo(
        device_idx=_GPU_INDEX,
        ckpt='last.ckpt',
        config='configs/sd-objaverse-finetune-c_concat-256.yaml'):

    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    demo = gr.Blocks(title=_TITLE)

    with demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column():

                image_block = gr.Image(type='pil', image_mode='RGBA',
                                       label='Input image of single object')
                preprocess_chk = gr.Checkbox(
                    True, label='Preprocess image automatically (remove background and recenter object)')
                # info='If enabled, the uploaded image will be preprocessed to remove the background and recenter the object by cropping and/or padding as necessary. '
                # 'If disabled, the image will be used as-is, *BUT* a fully transparent or white background is required.'),

                polar_slider = gr.Slider(
                    -90, 90, value=0, step=5, label='Polar angle (vertical rotation in degrees)')
                # info='Positive values move the camera down, while negative values move the camera up.')
                azimuth_slider = gr.Slider(
                    -180, 180, value=0, step=5, label='Azimuth angle (horizontal rotation in degrees)')
                # info='Positive values move the camera right, while negative values move the camera left.')
                radius_slider = gr.Slider(
                    -0.5, 0.5, value=0.0, step=0.1, label='Radius offset (distance from center)')
                # info='Positive values move the camera further away, while negative values move the camera closer.')

                scale_slider = gr.Slider(0, 30, value=3, step=1, label='cfg scale')
                samples_slider = gr.Slider(1, 8, value=4, step=1,
                                           label='Number of samples to generate')
                steps_slider = gr.Slider(5, 200, value=100, step=5,
                                         label='Number of diffusion steps')

                with gr.Row():
                    vis_btn = gr.Button('Visualize Angles')
                    run_btn = gr.Button('Run Generation')

            with gr.Column():

                vis_output = gr.Plot(
                    label='Relationship between input (green) and output (blue) camera pose')
                # vis_output = gr.Markdown('lol')
                gen_output = gr.Gallery(label='Generated images from specified new viewpoint')
                gen_output.style(grid=2)
                preproc_output = gr.Image(type='pil', image_mode='RGB',
                                          label='Preprocessed input image', visible=_SHOW_INTERMEDIATE)

        gr.Markdown(_ARTICLE)

        cam_vis = CameraVisualizer(vis_output)
        # polar_slider.change(fn=cam_vis.polar_change, inputs=polar_slider, outputs=vis_output)
        # azimuth_slider.change(fn=cam_vis.azimuth_change, inputs=azimuth_slider, outputs=vis_output)
        # radius_slider.change(fn=cam_vis.radius_change, inputs=radius_slider, outputs=vis_output)

        vis_btn.click(fn=partial(main_run, model, device, cam_vis, True),
                      inputs=[image_block, preprocess_chk, polar_slider, azimuth_slider,
                              radius_slider],
                      outputs=[vis_output, preproc_output])

        run_btn.click(fn=partial(main_run, model, device, cam_vis, False),
                      inputs=[image_block, preprocess_chk, polar_slider, azimuth_slider,
                              radius_slider, scale_slider, samples_slider, steps_slider],
                      outputs=[vis_output, gen_output, preproc_output])

    demo.launch(enable_queue=True, share=True)


if __name__ == '__main__':

    fire.Fire(run_demo)
