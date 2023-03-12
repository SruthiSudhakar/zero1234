Installation: 

```
conda create -n zero123 python==3.9
conda activate zero123  
cd stable-diffusion
pip install -r /requirements.txt  
git clone https://github.com/CompVis/taming-transformers.git  
pip install -e taming-transformers/  
git clone https://github.com/openai/CLIP.git  
pip install -e CLIP/ 
```

Run gradio demo: 

```
python gradio_objaverse.py
```
SJC:

```
cd sjc
pip install -r requirements.txt
python run_sjc_objaverse.py \
    --scene 'pikachu' \
    --index 0 \
    --n_steps 10000 \
    --lr 0.05 \
    --sd.scale 100.0 \
    --emptiness_weight 10000 \
    --emptiness_step 0.5 \
    --emptiness_scale 10 \
    --emptiness_multiplier 20. \
    --depth_smooth_weight 10000. \
    --depth_weight 0 \
    --view_weight 10000 \
    --train_view True \
    --prefix 'experiments/exp_wild' \
    --vox.blend_bg_texture False \
    --nerf_path '/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_wild'
```