from contextlib import nullcontext
from functools import partial

import math
import fire
import gradio as gr
import numpy as np
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
from PIL import Image, ImageDraw
import cv2
import os
import time
from ldm.util import load_and_preprocess


from scripts.image_objaverse import load_model_from_config
from datetime import datetime
from uuid import uuid4


def save_video(imgs, path, frame_rate=30):
    videodims = (256, 256)
    fourcc = cv2.VideoWriter_fourcc(*'vp09')    
    video = cv2.VideoWriter(path, fourcc, frame_rate, videodims)
    for i in range(0, len(imgs)):
        imtemp = imgs[i].copy()
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, \
                 ddim_eta, T):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples,1,1)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()\
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

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main(
    model,
    device,
    input_im,
    x=0.,
    z=0.,
    batch_size=16,
    total_views=64,
    frame_rate=30,
    scale=3.0,
    ddim_steps=50,
    preprocess=True,
    ddim_eta=1.0,
    precision="fp32",
    h=256,
    w=256,
    ):
    # input_im[input_im == [0., 0., 0.]] = [1., 1., 1., 1.]

    print(input_im.size)
    if preprocess:
        input_im = load_and_preprocess(input_im)
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.
        input_im[input_im[:, :, -1] == 0.] = [1., 1., 1., 1.]
        input_im = input_im[:, :, :3]
    print(input_im.shape)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(model)

    N_batches = total_views // batch_size

    # setup T vector for conditining
    T = torch.zeros(total_views, 1, 4)
    T[:, :, 0] = math.radians(x)
    T[:, :, 3] = math.radians(z)
    for i_view in range(total_views):
        y = 360. / total_views * i_view
        # print(y)
        T[i_view, :, 1] = math.sin(math.radians(y))
        T[i_view, :, 2] = math.cos(math.radians(y))

    output_ims = []
    for i_batch in range(N_batches):
        T_batch = T[i_batch * batch_size : (i_batch + 1) * batch_size].to(device)

        x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w,\
                                      ddim_steps, batch_size, scale, ddim_eta, T_batch)
        for x_sample in x_samples_ddim:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    video_id = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()) + '.webm'
    video_path = os.path.join('/home/rliu/Desktop/cvfiler04/ruoshi/github/stable-diffusion/inference_examples/videos', video_id)

    save_video(output_ims, video_path, frame_rate=frame_rate)
    time.sleep(1)

    return video_path


# description = \
# """Generate variations on an input image using a fine-tuned version of Stable Diffision.
# Trained by [Justin Pinkney](https://www.justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)) at [Lambda](https://lambdalabs.com/)

# __Get the [code](https://github.com/justinpinkney/stable-diffusion) and [model](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned).__

# ![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)

# """

# article = \
# """
# ## How does this work?

# The normal Stable Diffusion model is trained to be conditioned on text input. This version has had the original text encoder (from CLIP) removed, and replaced with
# the CLIP _image_ encoder instead. So instead of generating images based a text input, images are generated to match CLIP's embedding of the image.
# This creates images which have the same rough style and content, but different details, in particular the composition is generally quite different.
# This is a totally different approach to the img2img script of the original Stable Diffusion and gives very different results.

# The model was fine tuned on the [LAION aethetics v2 6+ dataset](https://laion.ai/blog/laion-aesthetics/) to accept the new conditioning.
# Training was done on 4xA6000 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
# More details on the method and training will come in a future blog post.
# """


def run_demo(
    device_idx=0,
    ckpt="logs/2023-02-23T06-42-58_sd-objaverse-finetune-c_concat-256/checkpoints/last.ckpt",
    # ckpt="logs/2023-02-15T21-34-54_sd-objaverse-finetune/checkpoints/last.ckpt",
    config="configs/stable-diffusion/sd-objaverse-finetune-c_concat-256.yaml",
    ):

    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    inputs = [
        gr.Image(type="pil", image_mode="RGBA", shape=[256, 256]),
        gr.Number(label="polar (between axis z+)"),
        gr.Number(label="z (distance from center)"),
        gr.Slider(0, 64, value=32, step=1, label="batch size"),
        gr.Slider(32, 512, value=32, step=16, label="total frames (should be multiple of batch size)"),
        gr.Slider(0, 100, value=3, step=1, label="cfg scale"),
        gr.Slider(1, 8, value=4, step=1, label="Number images"),
        gr.Slider(5, 200, value=100, step=5, label="steps"),
        gr.Checkbox(False, label="image preprocess (background removal and recenter)"),
    ]
    output = gr.Video(label="Generated 360 views")
    # output.style(grid=2)

    fn_with_model = partial(main, model, device)
    fn_with_model.__name__ = "fn_with_model"

    # examples = [
    #     ["assets/im-examples/vermeer.jpg", 3, 1, True, 25],
    #     ["assets/im-examples/matisse.jpg", 3, 1, True, 25],
    # ]

    demo = gr.Interface(
        fn=fn_with_model,
        title="Stable Diffusion Novel View Synthesis (Video)",
        # description=description,
        # article=article,
        inputs=inputs,
        outputs=output,
        # examples=examples,
        allow_flagging="never",
        )
    demo.launch(enable_queue=True, share=True)

if __name__ == "__main__":
    fire.Fire(run_demo)
