
CUDA_VISIBLE_DEVICES=1 \
python run_sjc_objaverse.py \
--scene 'alma' \
--index 0 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 0.5 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth True \
--train_normal False \
--train_view True \
--prefix 'experiments/exp_wild' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_wild"
