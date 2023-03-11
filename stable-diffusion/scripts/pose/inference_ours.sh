CUDA_VISIBLE_DEVICES=0 python scripts/inference_ours.py \
	--ckpt '/home/rliu/Desktop/65000.ckpt' \
	--config 'configs/stable-diffusion/sd-objaverse-finetune-c_concat-256.yaml' \
	--dataset 'RTMV' \
	--seed 0 \
	--steps 100 \
	--scale 10.0

CUDA_VISIBLE_DEVICES=1 python scripts/inference_ours.py \
	--ckpt '/home/rliu/Desktop/65000.ckpt' \
	--config 'configs/stable-diffusion/sd-objaverse-finetune-c_concat-256.yaml' \
	--dataset 'RTMV' \
	--seed 0 \
	--steps 100 \
	--scale 3.0

CUDA_VISIBLE_DEVICES=0 python scripts/inference_ours.py \
	--ckpt '/home/rliu/Desktop/80000.ckpt' \
	--config 'configs/stable-diffusion/sd-objaverse-finetune-c_concat-256.yaml' \
	--dataset 'GSO' \
	--split 'test' \
	--seed 0 \
	--steps 100 \
	--scale 10.0

CUDA_VISIBLE_DEVICES=1 python scripts/inference_ours.py \
	--ckpt '/home/rliu/Desktop/80000.ckpt' \
	--config 'configs/stable-diffusion/sd-objaverse-finetune-c_concat-256.yaml' \
	--dataset 'GSO' \
	--split 'test' \
	--seed 0 \
	--steps 100 \
	--scale 3.0


CUDA_VISIBLE_DEVICES=0 python scripts/inference_ours.py \
	--ckpt '/home/rliu/Desktop/80000.ckpt' \
	--config 'configs/stable-diffusion/sd-objaverse-finetune-c_concat-256.yaml' \
	--dataset 'WILD' \
	--seed 0 \
	--steps 100 \
	--scale 6.0

CUDA_VISIBLE_DEVICES=1 python scripts/inference_ours.py \
	--ckpt '/home/rliu/Desktop/80000.ckpt' \
	--config 'configs/stable-diffusion/sd-objaverse-finetune-c_concat-256.yaml' \
	--dataset 'WILD' \
	--seed 0 \
	--steps 100 \
	--scale 3.0