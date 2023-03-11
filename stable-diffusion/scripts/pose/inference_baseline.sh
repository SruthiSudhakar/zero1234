
CUDA_VISIBLE_DEVICES=0 python scripts/inference_variations.py \
	--dataset 'RTMV' \
	--seed 0 \
	--steps 100 \
	--scale 3.0

CUDA_VISIBLE_DEVICES=1 python scripts/inference_variations.py \
	--dataset 'GSO' \
	--split 'test' \
	--seed 0 \
	--steps 100 \
	--scale 3.0