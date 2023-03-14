

## Usage

Installation:

```
conda create -n zero123 python==3.9
conda activate zero123
cd stable-diffusion
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

Download checkpoint under `stable-diffusion`:

```
https://drive.google.com/file/d/17ACfBpqhO5o83-ZF8HUCnz5soa4m5jF1/view?usp=sharing
```

Run our gradio demo for novel view synthesis:

```
python gradio_new.py
```

Note that this app uses around 29 GB of VRAM, so it may not be possible to run it on any GPU.

Apply SJC for 3D reconstruction:

```
cd sjc
pip install -r requirements.txt
python run_sjc_objaverse.py \
    --scene 'pikachu' \
    --index 0 \
    --n_steps 10000 \
    --lr 0.05 \
    --sd.scale 100.0 \
    --depth_smooth_weight 10000. \
    --near_view_weight 10000. \
    --train_view True \
    --prefix 'experiments/exp_wild' \
    --vox.blend_bg_texture False \
    --nerf_path 'data/nerf_wild'
```
