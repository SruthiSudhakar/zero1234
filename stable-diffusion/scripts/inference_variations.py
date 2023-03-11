from ldm.data.nerf_like import *

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
import math
from PIL import Image


from scripts.image_variations import load_model_from_config
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed: int):
	import random, os
	import numpy as np
	import torch
	
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument(
	"--ckpt",
	type=str,
	nargs="?",
	default="/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/stable-diffusion/models/ldm/stable-diffusion-v1/sd-clip-vit-l14-img-embed_ema_only.ckpt",
	help="path to checkpoint to load model state from"
)
parser.add_argument(
	"--name",
	type=str,
	const=True,
	default="",
	nargs="?",
	help="logdir for results",
)
parser.add_argument(
	"--config",
	type=str,
	nargs="?",
	default="configs/stable-diffusion/sd-image-condition-finetune.yaml",
	help="path to configuration file"
)
parser.add_argument(
	"--dataset",
	type=str,
	nargs="?",
	default="",
	help="choices: ['RTMV', 'GSO']"
)
parser.add_argument(
	"--split",
	type=str,
	nargs="?",
	default="",
	help="required for GSO"
)
parser.add_argument(
	"--seed",
	type=int,
	nargs="?",
	default=0,
	help="random seed"
)
parser.add_argument(
	"--steps",
	type=int,
	nargs="?",
	default=100,
	help="ddim sampling steps"
)
parser.add_argument(
	"--scale",
	type=float,
	nargs="?",
	default=3.0,
	help="cfg guidance scale"
)

args = parser.parse_args()

seed_everything(args.seed)
device = "cuda:0"
config = OmegaConf.load(args.config)
model = load_model_from_config(config, args.ckpt, device=device)
sampler = DDIMSampler(model)

if args.dataset == 'RTMV':
	dataset = RTMV(first_K=17, resolution=256, load_target=False)
	batch_size = 8
	args.split = 'test' # no val split for RTMV
else:
	dataset = GSO(split=args.split, resolution=256, load_target=False)
	batch_size = 4

for i_scene, scene in enumerate(dataset.scene_list):
	print('=== inference for scene %s of dataset %s ===' % (scene, args.dataset))
	input_im, _ = dataset[i_scene]

	input_im = input_im
	input_im = input_im.to(device)

	precision_scope = nullcontext
	save_path = os.path.join('/home/rliu/Desktop/cvfiler04/ruoshi/github/stable-diffusion/iccv-results/variations', args.dataset)
	result_path = os.path.join(save_path, args.split, 'seed-%d_step-%d_scale-%.1f' % (args.seed, args.steps, args.scale))
	os.makedirs(result_path, exist_ok = True)
	with precision_scope("cuda"):
		with model.ema_scope():
			c = model.get_learned_conditioning(input_im).tile(batch_size, 1, 1)

			if args.scale != 1.0:
				uc = torch.zeros_like(c)
			else:
				uc = None

			samples_ddim, _ = sampler.sample(S=args.steps,
								 conditioning=c,
								 batch_size=batch_size,
								 shape=[4, 64, 64],
								 verbose=False,
								 unconditional_guidance_scale=args.scale,
								 unconditional_conditioning=uc,
								 eta=1.0,
								 x_T=None)

			x_samples_ddim = model.decode_first_stage(samples_ddim)
			imgs = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu().permute(0, 2, 3, 1).numpy()
			imgs = (imgs * 255.).astype(np.uint8)
			scene_path = os.path.join(result_path, scene)
			os.makedirs(scene_path, exist_ok = True)
			for i_img in range(len(imgs)):
				img_path = os.path.join(scene_path, '%d.png' % (i_img + 1)) # first image used as input
				img = Image.fromarray(imgs[i_img])
				img.save(img_path)

			# save input image as well
			input_im = torch.clamp((input_im + 1.0) / 2.0, min=0.0, max=1.0).cpu().permute(0, 2, 3, 1).numpy()
			input_im = (input_im * 255.).astype(np.uint8)
			img_input = Image.fromarray(input_im[0])
			img_input.save(os.path.join(scene_path, 'input.png'))