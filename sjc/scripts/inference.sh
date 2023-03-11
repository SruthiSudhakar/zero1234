CUDA_VISIBLE_DEVICES=1 \
python run_sjc_3d.py \
--train_view True \
--scene 'plane' \
--index 6 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 20.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth True \
--train_normal False \
--train_view False \
--prefix 'experiments/cache' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_shapenet" &

CUDA_VISIBLE_DEVICES=4 \
python run_sjc_3d.py \
--train_view True \
--scene 'camera' \
--index 6 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 20.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth True \
--train_normal False \
--train_view False \
--prefix 'exp_shapenet' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_shapenet" &

CUDA_VISIBLE_DEVICES=1 \
python run_sjc_3d.py \
--train_view True \
--scene 'chair' \
--index 6 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 100.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth True \
--train_normal False \
--train_view False \
--pose.up y \
--prefix 'exp_shapenet_pointe' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_shapenet" &

CUDA_VISIBLE_DEVICES=0 \
python run_sjc_3d.py \
--train_view True \
--scene 'headphone' \
--index 6 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 20.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth True \
--train_normal False \
--train_view False \
--prefix 'exp_shapenet' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_shapenet" &

CUDA_VISIBLE_DEVICES=4 \
python run_sjc_3d.py \
--train_view True \
--scene 'piano' \
--index 6 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 20.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth True \
--train_normal False \
--train_view False \
--prefix 'exp_shapenet' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_shapenet" &

CUDA_VISIBLE_DEVICES=6 \
python run_sjc_3d.py \
--train_view True \
--scene 'lamp' \
--index 6 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 20.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth True \
--train_normal False \
--train_view False \
--prefix 'exp_shapenet' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_shapenet" &