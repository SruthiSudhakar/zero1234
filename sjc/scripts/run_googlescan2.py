import os
import json

DATA_ROOT = "/home/rliu/cvfiler04/datasets/GoogleScannedObjects"
# path = os.path.join(DATA_ROOT, "val.json")
path = os.path.join(DATA_ROOT, "test.json")

with open(path, 'r') as f:
    filenames = json.load(f)[:20]
filenames = filenames[10:]
print(filenames)

for name in filenames:
    print(name)

    cmd = f"CUDA_VISIBLE_DEVICES=1 \
    python run_sjc_googlescan.py \
    --scene '{name}' \
    --index 0 \
    --n_steps 10000 \
    --lr 0.05 \
    --sd.scale 100.0 \
    --emptiness_weight 5000 \
    --emptiness_step 0.5 \
    --emptiness_scale 10 \
    --emptiness_multiplier 20. \
    --depth_smooth_weight 5000. \
    --depth_weight 0 \
    --view_weight 10000 \
    --train_depth False \
    --train_normal False \
    --train_view True \
    --prefix 'experiments/googlescan_test_new' \
    --vox.blend_bg_texture False"
    
    os.system(cmd)
