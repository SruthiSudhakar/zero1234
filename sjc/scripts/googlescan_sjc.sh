CUDA_VISIBLE_DEVICES=0 \
python run_sjc_baseline_gso.py \
--scene 'Sonny_School_Bus' \
--index 0 \
--n_steps 1 \
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
--prefix 'experiments/cache' \
--vox.blend_bg_texture False \
--nerf_path "/home/rliu/Desktop/cvfiler04/datasets/GoogleScannedObjects"


# CUDA_VISIBLE_DEVICES=5 \
# python load_evaluate.py \
# --scene 'Sonny_School_Bus' \
# --index 0 \
# --n_steps 10000 \
# --lr 0.05 \
# --sd.scale 100.0 \
# --emptiness_weight 5000 \
# --emptiness_step 0.5 \
# --emptiness_scale 10 \
# --emptiness_multiplier 20. \
# --depth_smooth_weight 5000. \
# --depth_weight 0 \
# --view_weight 10000 \
# --train_depth False \
# --train_normal False \
# --train_view True \
# --prefix 'experiments/googlescan_val_new' \
# --vox.blend_bg_texture False