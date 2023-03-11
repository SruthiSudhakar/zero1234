# CUDA_VISIBLE_DEVICES=0 \
# python run_sjc_googlescan.py \
# --scene 'Toysmith_Windem_Up_Flippin_Animals_Dog' \
# --index 6 \
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
# --prefix 'experiments/googlescan_test' \
# --vox.blend_bg_texture False


# CUDA_VISIBLE_DEVICES=7 \
# python run_sjc_googlescan.py \
# --scene 'SPEED_BOAT' \
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
# --prefix 'experiments/googlescan_test' \
# --vox.blend_bg_texture False &


# CUDA_VISIBLE_DEVICES=6 \
# python run_sjc_googlescan.py \
# --scene 'TOOL_BELT' \
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
# --prefix 'experiments/googlescan_test' \
# --vox.blend_bg_texture False &


# CUDA_VISIBLE_DEVICES=5 \
# python run_sjc_googlescan.py \
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
# --prefix 'experiments/googlescan_test' \
# --vox.blend_bg_texture False

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


CUDA_VISIBLE_DEVICES=1 \
python run_sjc_googlescan.py \
--scene 'Sonny_School_Bus' \
--index 0 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 50.0 \
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
--prefix 'experiments/googlescan_test' \
--vox.blend_bg_texture False

CUDA_VISIBLE_DEVICES=0 \
python run_sjc_googlescan.py \
--scene '00001' \
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
--prefix 'experiments/rtmv' \
--vox.blend_bg_texture False
