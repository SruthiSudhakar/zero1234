CUDA_VISIBLE_DEVICES=0 \
python run_sjc_pointe.py \
--train_view False \
--scene 'airpurifier' \
--index 2 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 100.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth False \
--train_normal False \
--train_view False  \
--pose.up z \
--prefix 'exp_shapenet_pointe' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_synthetic" \
--nerf_split "train" \
--voxel_size 0.1 &

CUDA_VISIBLE_DEVICES=1 \
python run_sjc_pointe.py \
--train_view False \
--scene 'toiletpaper' \
--index 2 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 100.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth False \
--train_normal False \
--train_view False  \
--pose.up z \
--prefix 'exp_shapenet_pointe' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_synthetic" \
--nerf_split "train" \
--voxel_size 0.1 &

CUDA_VISIBLE_DEVICES=2 \
python run_sjc_pointe.py \
--train_view False \
--scene 'bikehat' \
--index 2 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 100.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth False \
--train_normal False \
--train_view False  \
--pose.up z \
--prefix 'exp_shapenet_pointe' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_synthetic" \
--nerf_split "train" \
--voxel_size 0.1 &

CUDA_VISIBLE_DEVICES=3 \
python run_sjc_pointe.py \
--train_view False \
--scene 'trashbin' \
--index 2 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 100.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth False \
--train_normal False \
--train_view False  \
--pose.up z \
--prefix 'exp_shapenet_pointe' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_synthetic" \
--nerf_split "train" \
--voxel_size 0.1 &

CUDA_VISIBLE_DEVICES=4 \
python run_sjc_pointe.py \
--train_view False \
--scene 'headset' \
--index 2 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 100.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth False \
--train_normal False \
--train_view False  \
--pose.up z \
--prefix 'exp_shapenet_pointe' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_synthetic" \
--nerf_split "train" \
--voxel_size 0.1 &

CUDA_VISIBLE_DEVICES=5 \
python run_sjc_pointe.py \
--train_view False \
--scene 'mouse' \
--index 2 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 100.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth False \
--train_normal False \
--train_view False  \
--pose.up z \
--prefix 'exp_shapenet_pointe' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_synthetic" \
--nerf_split "train" \
--voxel_size 0.1 &

CUDA_VISIBLE_DEVICES=6 \
python run_sjc_pointe.py \
--train_view False \
--scene 'hermanmiller' \
--index 2 \
--n_steps 10000 \
--lr 0.05 \
--sd.scale 100.0 \
--emptiness_weight 10000 \
--emptiness_step 0.5 \
--emptiness_multiplier 100.0 \
--depth_weight 0 \
--view_weight 10000 \
--train_depth False \
--train_normal False \
--train_view False  \
--pose.up z \
--prefix 'exp_shapenet_pointe' \
--nerf_path "/home/rliu/Desktop/cvfiler04/ruoshi/github/sjc/data/nerf_synthetic" \
--nerf_split "train" \
--voxel_size 0.1 &
