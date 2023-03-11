# finetune image variation
TORCH_DISTRIBUTED_DEBUG=DETAIL python main.py \
    -t \
    --base configs/stable-diffusion/sd-image-condition-finetune.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from models/ldm/sd-image-conditioned-v2.ckpt

python main.py \
    -t \
    --base configs/stable-diffusion/sd-image-condition-finetune.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from models/ldm/sd-image-conditioned-v2.ckpt

# finetune objaverse
python main.py \
    -t \
    --base configs/stable-diffusion/sd-objaverse-finetune.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from models/ldm/sd-image-conditioned-v2.ckpt


python main.py \
    -t \
    --base configs/stable-diffusion/sd-objaverse-finetune-c_concat-256.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --resume logs/2023-02-23T06-42-58_sd-objaverse-finetune-c_concat-256/checkpoints/last.ckpt


python main.py \
    -t \
    --base configs/stable-diffusion/sd-objaverse-finetune-c_concat.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from models/ldm/sd-image-conditioned-v2.ckpt
