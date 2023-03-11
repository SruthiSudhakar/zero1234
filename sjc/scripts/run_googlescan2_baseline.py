import os
import json

DATA_ROOT = "/home/rliu/Desktop/cvfiler04/datasets/GoogleScannedObjects"
# path = os.path.join(DATA_ROOT, "val.json")
path = os.path.join(DATA_ROOT, "test.json")

with open(path, 'r') as f:
    filenames = json.load(f)[:10]
# filenames = filenames[:10]
# filenames = ['Sonny_School_Bus', 'Vtech_Roll_Learn_Turtle']
print(filenames)

for name in filenames:
    print(name)

    cmd = f"CUDA_VISIBLE_DEVICES=0 \
    python run_sjc_baseline_gso.py \
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
    --prefix 'experiments/googlescan_test_new_baseline_25' \
    --vox.blend_bg_texture False \
    --nerf_path '/home/rliu/Desktop/cvfiler04/datasets/GoogleScannedObjects'"
    
    os.system(cmd)
