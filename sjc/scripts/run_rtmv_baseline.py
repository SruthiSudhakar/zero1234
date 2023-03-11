import os
import json

filenames = [f"0000{i}" for i in range(10)]
print(filenames)

for name in filenames:
    print(name)

    cmd = f"CUDA_VISIBLE_DEVICES=7 \
    python run_sjc_baseline_RTMV.py \
    --scene '{name}' \
    --index 0 \
    --n_steps 10000 \
    --lr 0.05 \
    --sd.scale 100.0 \
    --emptiness_weight 10000 \
    --emptiness_step 0.5 \
    --emptiness_multiplier 20.0 \
    --depth_weight 0 \
    --view_weight 10000 \
    --train_depth False \
    --train_normal False \
    --train_view True \
    --prefix 'experiments/rtmv' \
    --vox.blend_bg_texture False \
    --nerf_path '/home/rliu/Desktop/cvfiler04/datasets/RTMV/google_scanned'"
    
    os.system(cmd)