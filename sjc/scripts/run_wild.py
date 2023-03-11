import os
import json

DATA_ROOT = "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_wild"
# path = os.path.join(DATA_ROOT, "val.json")
filenames = os.listdir(DATA_ROOT)
print(filenames)

for name in filenames:
    print(name)

    cmd = f"CUDA_VISIBLE_DEVICES=1 \
    python run_sjc_objaverse.py \
    --scene '{name}' \
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
    --train_depth False \
    --train_normal False \
    --train_view True \
    --prefix 'experiments/exp_wild' \
    --vox.blend_bg_texture False \
    --nerf_path '/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_wild'"
    os.system(cmd)
